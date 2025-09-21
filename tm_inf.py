
# One-day VaR forecast using your fine-tuned TimesFM core on MPS.

from pathlib import Path
import numpy as np, pandas as pd, torch, timesfm
from tqdm import tqdm
from timesfm import TimesFmHparams, TimesFmCheckpoint

# ----- Paths / parameters -----

TRAIN_XLSX = "dataset/trainset.xlsx"
TEST_XLSX  = "dataset/testset.xlsx"
OUTPUT_CSV = Path("results/result_H1.csv")
CONTEXT_LEN = 512
HORIZON_LEN = 1
QUANTILES   = [0.01, 0.025, 0.05, 0.10, 0.50, 0.60, 0.70, 0.80, 0.90]
CKPT_REPO   = "google/timesfm-1.0-200m-pytorch"
FINETUNED_CORE = "finetuned_model/best_core_H1.pt"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ----- build hparams & model -----

hparams = TimesFmHparams(
    backend             = "mps" if DEVICE.type == "mps" else ("cuda" if DEVICE.type == "cuda" else "cpu"),
    horizon_len         = HORIZON_LEN,
    context_len         = CONTEXT_LEN,
    model_dims          = 1280,
    per_core_batch_size = 32,
    quantiles           = QUANTILES,
    input_patch_len     = 32,
    output_patch_len    = 128,
    num_layers          = 20,
)
tfm = timesfm.TimesFm(hparams=hparams, checkpoint=TimesFmCheckpoint(huggingface_repo_id=CKPT_REPO))

core = tfm._model
core.to(DEVICE).eval()
state = torch.load(FINETUNED_CORE, map_location=DEVICE)
core.load_state_dict(state, strict=False)
print("loaded fine-tuned core from", FINETUNED_CORE)

# ----- load returns -----

def read_sheet(xlsx: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx, header=0, index_col=0)
    df.index = pd.to_datetime(df.index, dayfirst=True)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

train_df = read_sheet(TRAIN_XLSX)
test_df  = read_sheet(TEST_XLSX)
common_cols = [c for c in train_df.columns if c in test_df.columns] or list(train_df.columns)
train_df = train_df[common_cols].copy()
test_df  = test_df[common_cols].copy()
full_df  = pd.concat([train_df, test_df]).sort_index()
assets   = full_df.columns.tolist()
data_np  = full_df.values.astype(np.float32)

# ----- device-native decode -----

# Here again, when it comes to shapes, I might be doing something wrong
@torch.no_grad()
def var_one_day_device(row_idx: int, mat: np.ndarray) -> np.ndarray:
    ctx = mat[row_idx - CONTEXT_LEN : row_idx]      # (L, N)
    x_ctx = torch.from_numpy(ctx.T.copy()).to(DEVICE)  # (B=N, L)
    B, L = x_ctx.shape
    assert L == CONTEXT_LEN, f"expected L={CONTEXT_LEN}, got {L}"
    x_pad = torch.zeros(B, CONTEXT_LEN + HORIZON_LEN, dtype=torch.float32, device=DEVICE)
    # IMPORTANT: shape (B, 1), not (B,)
    freq  = torch.zeros(B, 1, dtype=torch.long, device=DEVICE)

    mean_out, q_out = core.decode(x_ctx, x_pad, freq, horizon_len=HORIZON_LEN)
    # q_out: (B, H, Q_total). Some builds may include an extra channel; slice to your quantiles:
    q_out = q_out[:, :, :len(QUANTILES)]
    return q_out[:, 0, :].detach().cpu().numpy()  # (B, |Q|)


# ----- roll over test period -----

records   = {q: [] for q in QUANTILES}
start_idx = full_df.index.get_loc(test_df.index[0])

print(f"[INFO] forecasting with fine-tuned core on {DEVICE.type} …")
for ridx, day in tqdm(enumerate(test_df.index, start=start_idx), total=len(test_df)):
    qmat = var_one_day_device(ridx, data_np)  # (N, |Q|)
    for qi, q in enumerate(QUANTILES):
        records[q].append(pd.Series(qmat[:, qi], index=assets, name=day))

# ----- save -----

panels = []
for q, lst in records.items():
    df = pd.concat(lst, axis=1).T
    df.columns = pd.MultiIndex.from_product([[f"VaR_{q:.3f}"], df.columns])
    panels.append(df)

out = pd.concat(panels, axis=1).sort_index(axis=1)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT_CSV)
print(f"saved forecasts to {OUTPUT_CSV}  (rows={len(out)}, assets={len(assets)})")
