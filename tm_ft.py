from __future__ import annotations
import argparse, math, random, time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import timesfm
from timesfm import TimesFmHparams, TimesFmCheckpoint

# -------------------- Defaults / Paths --------------------
DEFAULT_TRAIN_XLSX = "dataset/trainset.xlsx"
DEFAULT_TEST_XLSX  = "dataset/testset.xlsx"    
OUTPUT_DIR         = Path("finetuned_model")
BEST_CORE_PATH     = OUTPUT_DIR / "best_core_H1.pt"
BEST_CORE_EMA_PATH = OUTPUT_DIR / "best_core_ema_H1.pt"

# -------------------- Model / Training knobs --------------------
CONTEXT_LEN        = 512
HORIZON_LEN        = 1                 
INPUT_PATCH_LEN    = 32
OUTPUT_PATCH_LEN   = 128
MODEL_DIMS         = 1280
NUM_LAYERS         = 20
HF_REPO            = "google/timesfm-1.0-200m-pytorch"

QUANTILES          = [0.01, 0.025, 0.05, 0.10, 0.50, 0.60, 0.70, 0.80, 0.90]

# Split by date like colleagues
# (they do by percent in their code, but the borders are taken exactly from the paper)
TRAIN_START_DATE   = "2005-01-03"
TRAIN_END_DATE     = "2011-12-30"
VAL_START_DATE     = "2012-01-02"
VAL_END_DATE       = "2014-12-31"

# Train parameters
# (copied virtually verbatim from their code)
BATCH_SIZE         = 16
MAX_STEPS          = 40_000
LR_START           = 1e-3
LR_END             = 1e-4
WEIGHT_DECAY       = 0.0
ADAM_EPS           = 1e-7
CLIP_NORM          = 100.0
EMA_DECAY          = 0.9999
PATIENCE_VALCHECKS = 5
VAL_EVERY_STEPS    = 1000

# Copied seed from our colleagues
SEED               = 2024

# Setting workers to 0 shoudl enable dynamic multi-threading 
NUM_WORKERS        = 0

# -------------------- Utils --------------------
def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device() -> torch.device:
    if torch.backends.mps.is_available(): 
        print("[INFO] device: mps")
        return torch.device("mps")
    
    if torch.cuda.is_available():
        print(f"[INFO] device: cuda:{torch.cuda.current_device()}"); 
        return torch.device("cuda")
    
    print("[INFO] device: cpu")
    return torch.device("cpu")

# Read returns (same/similar way as I do for all other my scripts)
# Improtant:    here we repalce non trading days with NaN, but this shoudl be
#               the best way to handle this apparently.
def read_returns(xlsx_path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, header=0, index_col=0)
    df.index = pd.to_datetime(df.index, dayfirst=True, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.replace(0.0, np.nan)
    return df.astype(np.float32)

# Retruns training and validation sets
def align_and_split(train_df: pd.DataFrame,
                    test_df: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_cols = [c for c in train_df.columns if (test_df is None or c in test_df.columns)]
    if not common_cols: common_cols = list(train_df.columns)
    train_df = train_df[common_cols].copy()
    tr = train_df.loc[TRAIN_START_DATE:TRAIN_END_DATE].copy()
    va = train_df.loc[VAL_START_DATE:VAL_END_DATE].copy()
    keep = [c for c in tr.columns if c in va.columns]
    return tr[keep], va[keep]

# -------------------- Dataset (per-asset compressed timeline) --------------------

# Build samples by dropping NaNs **per asset**, then sliding windows on each asset's compressed series.
class CompressedPerAssetWindows(Dataset):

    def __init__(self, frame: pd.DataFrame, context_len: int, horizon_len: int):
        super().__init__()
        
        # Set horizon and context lengths
        self.L = context_len
        self.H = horizon_len

        # Stores names of our assets
        self.assets: List[str] = frame.columns.tolist()
        # An initially empty list that will be filled with per-asset compressed time series
        # (it will became a "list of lists")
        self.series: List[np.ndarray] = []

        # A single [asset_idx, start_idx] uniquely identifies one window (or where one window starts actually)
        # (self.index_map is just a list of all valid window starts for each eligible asset’s compressed series)
        # (i.e. for asset 0 it might be [0,0], [0,1], [0,2] etc.)
        # All training samples are len(self.index_map)
        self.index_map: List[Tuple[int,int]] = []  # (asset_idx, start_idx)

        # Here we actually construct this index_map
        for col in self.assets:
            s = frame[col].dropna().to_numpy(dtype=np.float32)
            if s.shape[0] >= (self.L + self.H):
                self.series.append(s)
                aidx = len(self.series) - 1
                # enumerate valid window starts on compressed series
                for start in range(0, s.shape[0] - (self.L + self.H) + 1):
                    self.index_map.append((aidx, start))

        self.n_samples = len(self.index_map)

    # Lets just check how many samples we have
    def __len__(self) -> int:
        return self.n_samples

    # Data getter
    # Given an integer idx, it builds one training example (one window) and 
    # returns a 4-tuple of tensors ready for TimesFM’s decode(...)

    def __getitem__(self, idx: int):
        # self.index_map[idx] tells which asset (compressed series) and where the window starts.
        # seq is that asset’s trading-only 1-D NumPy array (NaNs already removed).
        asset_idx, start = self.index_map[idx]
        seq = self.series[asset_idx]

        # ctx = the L past values the model will see.
        # fut = the H next values the model should forecast (supervision target).
        ctx = seq[start : start + self.L]                 # (L,)
        fut = seq[start + self.L : start + self.L + self.H]  # (H,)

        # Some conversions
        # (code was breaking here before so this was GPT recommended fix)
        x_ctx = torch.from_numpy(ctx.astype(np.float32))
        x_pad = torch.zeros(self.L + self.H, dtype=torch.float32)
        x_fut = torch.from_numpy(fut.astype(np.float32))
        freq  = torch.zeros(1, dtype=torch.long)          # (1,) daily
        return x_ctx, x_pad, freq, x_fut

# A factory that turns a returns DataFrame (dates × assets) into a PyTorch DataLoader
def make_loader(frame: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    # Builds the dataset from your frame:
    # Per asset: drop NaNs (0.0 were already turned into NaN), producing a compressed trading-only series.

    # Call to previously defined function
    ds = CompressedPerAssetWindows(frame, CONTEXT_LEN, HORIZON_LEN)

    # Just a quick safety check to see if there are no valid windows after per-asset NaN drop.
    if len(ds) == 0:
        raise RuntimeError(
            "No valid windows after per-asset NaN drop. "
            "No asset has at least CONTEXT_LEN+HORIZON_LEN non-NaN points in the chosen dates."
        )
    
    # Counts how many distinct assets actually contributed at least one window 
    # (by looking at the first element of each (asset_idx, start_idx) pair in index_map).
    eligible_assets = len({a for a,_ in ds.index_map})
    print(f"[INFO] Dataset built: samples={len(ds):,}  (assets eligible: {eligible_assets})")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=False, drop_last=shuffle)

# -------------------- Loss / EMA / Freezing --------------------
# Pinall loss function - I am not entirely sure I am doing this good.
def pinball_loss(q_pred: torch.Tensor, target: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    
    # q_pred: [B, H, Q_total] (from decode); slice to first len(quantiles)
    # target: [B, H]
    
    # q_pred has predicted quantiles for each example and horizon step: [batch, horizon, num_heads].

    Q = len(quantiles)
    q_pred = q_pred[:, :, :Q]
    y = target.unsqueeze(-1)  # [B,H,1]
    
    # calculating residual
    e = y - q_pred
    qs = torch.tensor(quantiles, dtype=torch.float32, device=q_pred.device).view(1,1,Q)
    return torch.maximum(qs * e, (qs - 1.0) * e).mean()


# Run the TimesFM decoder during training to get quantile predictions for the next H = horizon_len steps, 
# and return only the first want_q quantile heads you plan to supervise with the pinball loss.
def decode_quantiles_train(core, x_ctx, x_pad, freq, horizon_len: int, want_q: int) -> torch.Tensor:
    
    # Training-time decode: NO torch.no_grad() so autograd tracks graph.
    # Returns first `want_q` quantile heads: [B, H, want_q]
    
    mean_out, q_out = core.decode(x_ctx, x_pad, freq, horizon_len=horizon_len)
    if q_out is None:
        raise RuntimeError("decode() returned q_out=None")
    if q_out.size(-1) < want_q:
        raise RuntimeError(f"Decoder exposed only {q_out.size(-1)} quantile heads, need {want_q}.")
    return q_out[:, :, :want_q]

# Eval-time decode with no_grad for speed/memory.
@torch.no_grad()
def decode_quantiles_eval(core, x_ctx, x_pad, freq, horizon_len: int, want_q: int) -> torch.Tensor:
 
    # Calls TimesFM’s high-level decoder to get the next H = horizon_len steps
    # mean_out is the point forecast (for our VaR task we are actually not interested in that)
    # q_out are the quantiles
    mean_out, q_out = core.decode(x_ctx, x_pad, freq, horizon_len=horizon_len)
    if q_out is None:
        raise RuntimeError("decode() returned q_out=None")
    if q_out.size(-1) < want_q:
        raise RuntimeError(f"Decoder exposed only {q_out.size(-1)} quantile heads, need {want_q}.")
    return q_out[:, :, :want_q]

# Goal here is to implement a linear-probe-style fine-tune: freeze the large transformer stack 
# inside TimesFM and leave the output/horizon layers trainable so only the heads adapt to your data.
# (this is how I understood code from our colleagues, but I also tried to train other layers and
# results were much worse anyway)
def freeze_transformer_stack(core: torch.nn.Module) -> Tuple[int,int]:
    trainable = frozen = 0
    keys = set()
    for name, _ in core.named_modules():
        if "stacked_transformer_layer" in name: keys.add("stacked_transformer_layer")
        if "stacked_transformer" in name:       keys.add("stacked_transformer")
    for name, p in core.named_parameters():
        if any(k in name for k in keys):
            p.requires_grad = False; frozen += p.numel()
        else:
            p.requires_grad = True;  trainable += p.numel()
    if frozen == 0:
        print("[WARN] transformer stack not detected; nothing frozen.")
    return trainable, frozen

# Returns a multiplier for PyTorch’s LambdaLR that smoothly decays the learning 
# rate from lr_start to lr_end over max_steps using a cosine schedule.
def cosine_lambda(step: int, max_steps: int, lr_start: float, lr_end: float) -> float:
    if step >= max_steps: return lr_end / lr_start
    cos = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    return (lr_end / lr_start) + (1.0 - (lr_end / lr_start)) * cos


# THis block defines a tiny Exponential Moving Average (EMA) utility to smooth trainable weights over time, 
# plus a few @torch.no_grad() decorators to make certain methods run without tracking gradients - I am not
# sure if tracking gradients is good or bad but I added this feature eventually while compiling to make it
# hopefully more efficient.
class EMA:
    # Builds an EMA tracker for the trainable parameters only (p.requires_grad).
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}

    # Call this every training step after optimizer.step() and updates each EMA weight in-place:
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                if n not in self.shadow:
                    self.shadow[n] = p.detach().clone()
                else:
                    self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    # Temporarily swap the model’s live trainable weights with the EMA weights
    @torch.no_grad()
    def store_apply(self, model: torch.nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.detach().clone()
                if n in self.shadow: p.data.copy_(self.shadow[n].data)

    # Reverts the temporary swap made by store_apply.
    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        if hasattr(self, "backup"):
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.backup:
                    p.data.copy_(self.backup[n].data)
            del self.backup

# -------------------- Main --------------------
def main():

    # All these arguments are added when I was debugging to make it easier.
    # However, they are optional and by default initialized with constants.
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_xlsx", type=str, default=DEFAULT_TRAIN_XLSX)
    parser.add_argument("--test_xlsx",  type=str, default=DEFAULT_TEST_XLSX)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--horizon_len",type=int, default=HORIZON_LEN)
    parser.add_argument("--max_steps",  type=int, default=MAX_STEPS)
    parser.add_argument("--lr_start",   type=float, default=LR_START)
    parser.add_argument("--lr_end",     type=float, default=LR_END)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--adam_eps",   type=float, default=ADAM_EPS)
    parser.add_argument("--clip_norm",  type=float, default=CLIP_NORM)
    parser.add_argument("--ema_decay",  type=float, default=EMA_DECAY)
    parser.add_argument("--patience",   type=int, default=PATIENCE_VALCHECKS)
    parser.add_argument("--save_ema",   action="store_true")
    args = parser.parse_args()

    set_all_seeds(SEED)
    DEV = pick_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Load & split (0->NaN, per ARMA-GARCH) -----
    df_train_all = read_returns(args.train_xlsx)
    df_test_all  = read_returns(args.test_xlsx) if args.test_xlsx else None
    df_train, df_val = align_and_split(df_train_all, df_test_all)
    print(f"[INFO] Rows: train={len(df_train):,} | val={len(df_val):,} | assets={df_train.shape[1]}")

    # ----- DataLoaders (per-asset compressed windows) -----
    dl_train = make_loader(df_train, args.batch_size, shuffle=True)
    dl_val   = make_loader(df_val,   args.batch_size, shuffle=False)

    # ----- Model -----
    hparams = TimesFmHparams(
        backend = ("mps" if DEV.type == "mps" else ("cuda" if DEV.type == "cuda" else "cpu")),
        horizon_len      = args.horizon_len,
        context_len      = CONTEXT_LEN,
        model_dims       = MODEL_DIMS,
        per_core_batch_size = args.batch_size,
        quantiles        = QUANTILES,
        input_patch_len  = INPUT_PATCH_LEN,
        output_patch_len = OUTPUT_PATCH_LEN,
        num_layers       = NUM_LAYERS,
    )
    tfm = timesfm.TimesFm(hparams=hparams, checkpoint=TimesFmCheckpoint(huggingface_repo_id=HF_REPO))
    core = tfm._model.to(DEV)

    trainable, frozen = freeze_transformer_stack(core)
    print(f"[INFO] trainable params: {trainable/1e6:.3f}M | frozen: {frozen/1e6:.3f}M")

    opt = torch.optim.Adam((p for p in core.parameters() if p.requires_grad),
                           lr=args.lr_start, weight_decay=args.weight_decay, eps=args.adam_eps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: cosine_lambda(s, args.max_steps, args.lr_start, args.lr_end))
    ema = EMA(core, args.ema_decay)

    # Sanity decode check (use eval helper here, not training)
    with torch.no_grad():
        xb = next(iter(dl_train))
        x_ctx, x_pad, freq, x_fut = [t.to(DEV) for t in xb]
        q = decode_quantiles_eval(core, x_ctx.float(), x_pad.float(), freq, horizon_len=args.horizon_len, want_q=len(QUANTILES))
        print(f"[DEBUG] decode q_out: {tuple(q.shape)} ; target: {tuple(x_fut.shape)}")

    # ----- Train loop -----
    best_val = float("inf"); best_step = 0; patience_left = args.patience
    global_step = 0; t0 = time.time()

    for epoch in range(1, 1000):  # ES will break earlier
        core.train()
        pbar = tqdm(dl_train, desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            global_step += 1
            x_ctx, x_pad, freq, x_fut = [t.to(DEV) for t in batch]
            # TRAIN-TIME decode (requires grad)
            q_pred = decode_quantiles_train(core, x_ctx.float(), x_pad.float(), freq, args.horizon_len, want_q=len(QUANTILES))
            loss = pinball_loss(q_pred, x_fut.float(), QUANTILES)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_((p for p in core.parameters() if p.requires_grad), max_norm=args.clip_norm)
            opt.step(); sched.step(); ema.update(core)

            if global_step % 200 == 0:
                pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

            if global_step % VAL_EVERY_STEPS == 0 or global_step >= args.max_steps:
                # Validate with EMA weights applied
                core.eval(); ema.store_apply(core)
                vlosses = []
                with torch.no_grad():
                    for vb in dl_val:
                        v_ctx, v_pad, v_freq, v_fut = [t.to(DEV) for t in vb]
                        v_q = decode_quantiles_eval(core, v_ctx.float(), v_pad.float(), v_freq, args.horizon_len, want_q=len(QUANTILES))
                        vlosses.append(pinball_loss(v_q, v_fut.float(), QUANTILES).item())
                ema.restore(core)
                vmean = float(np.mean(vlosses)) if vlosses else float("inf")
                print(f"[VAL] step {global_step} pinball={vmean:.6f} (best={best_val:.6f})")

                if vmean < best_val:
                    best_val = vmean; best_step = global_step; patience_left = args.patience
                    torch.save(core.state_dict(), BEST_CORE_PATH)
                    # Save EMA snapshot too
                    ema.store_apply(core); torch.save(core.state_dict(), BEST_CORE_EMA_PATH); ema.restore(core)
                    print(f"saved best core -> {BEST_CORE_PATH}  (EMA -> {BEST_CORE_EMA_PATH})")
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        elapsed = (time.time() - t0) / 60
                        print(f"Early stopping. Best val={best_val:.6f} @ step={best_step}. Elapsed {elapsed:.1f} min.")
                        print(f"Saved: {BEST_CORE_PATH} (and EMA {BEST_CORE_EMA_PATH})")
                        return

            if global_step >= args.max_steps:
                elapsed = (time.time() - t0) / 60
                print(f"Reached MAX_STEPS={args.max_steps}. Best val={best_val:.6f} @ step={best_step}. Elapsed {elapsed:.1f} min.")
                print(f"Saved: {BEST_CORE_PATH} (and EMA {BEST_CORE_EMA_PATH})")
                return

if __name__ == "__main__":
    main()