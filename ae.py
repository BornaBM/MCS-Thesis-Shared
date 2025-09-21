import pandas as pd
import numpy as np

df_returns = pd.read_excel("dataset/testset.xlsx", sheet_name="TEST SET")

df_returns.rename(columns={"Name": "Date"}, inplace=True)  # if your column is named 'Name'
df_returns["Date"] = pd.to_datetime(df_returns["Date"])

df_var = pd.read_csv("tm_splitted/VaR_0100.csv", index_col=0)
alp=0.1

df_var.reset_index(inplace=True)
df_var.rename(columns={"date": "Date"}, inplace=True)

df_var["Date"] = pd.to_datetime(df_var["Date"])

df_merged = pd.merge(
    df_returns, df_var,
    on="Date",
    how="inner",  # or "left" if you want all testset days
    suffixes=("_ret", "_var")
)

def compute_abs_one_minus_AE_skip0(df, return_col, var_col, alpha=0.01):

    # Extract just these two columns
    sub_df = df[[return_col, var_col]].copy()
    
    # Drop rows where either is NaN
    sub_df.dropna(inplace=True)
    
    # Drop rows where return == 0.0
    sub_df = sub_df[sub_df[return_col] != 0.0]
    
    total_days = len(sub_df)
    if total_days == 0:
        return np.nan  # no valid coverage data
    
    actual_violations = (sub_df[return_col] < sub_df[var_col]).sum()
    expected_violations = alpha * total_days
    if expected_violations == 0:
        return np.nan
    
    AE = actual_violations / expected_violations
    # added
    #violation_rate = actual_violations / total_days
    #print(f"Asset: {return_col[:-4]}, Violations: {actual_violations}, Days: {total_days}, AE: {AE:.3f}, Violation rate: {violation_rate:.3f}")
    # added

    return abs(1 - AE)


assets = []
for col in df_merged.columns:
    if col.endswith("_ret") and col != "Date_ret":
        # e.g. "AAPL_ret" => asset name "AAPL"
        asset_name = col[:-4]
        assets.append(asset_name)

results = []
for asset in assets:
    ret_col = asset + "_ret"
    var_col = asset + "_var"
    val = compute_abs_one_minus_AE_skip0(df_merged, ret_col, var_col, alpha=alp)#0.01)
    results.append((asset, val))

df_results = pd.DataFrame(results, columns=["Asset", "|1 - AE|"])

metric_vals = df_results["|1 - AE|"].dropna().values
if len(metric_vals) > 0:
    min_val    = metric_vals.min()
    max_val    = metric_vals.max()
    mean_val   = metric_vals.mean()
    median_val = np.median(metric_vals)
    std_val    = metric_vals.std()

    print("===== Rolling Hist VaR, skipping no-trade days => Coverage |1 - AE| =====")
    print(f"Number of Assets: {len(metric_vals)} (some might be NaN if zero data).")
    print(f"Min      : {min_val:.4f}")
    print(f"Mean     : {mean_val:.4f}")
    print(f"Median   : {median_val:.4f}")
    print(f"Max      : {max_val:.4f}")
    print(f"Std Dev  : {std_val:.4f}")
    print("=========================================================================")
else:
    print("[INFO] No valid coverage data after skipping 0.0 or NaN returns. Check dataset.")