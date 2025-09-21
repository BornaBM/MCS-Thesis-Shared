import pandas as pd
import os

# Load results
df_raw = pd.read_csv("results/result_H1.csv", header=None)


# Extract metadata
var_levels_row = df_raw.iloc[0]
stock_names_row = df_raw.iloc[1]
data = df_raw.iloc[2:].copy()

# Prepare output directory
output_folder = "tm_splitted"
os.makedirs(output_folder, exist_ok=True)

# Extract the date column
date_column = df_raw.iloc[2:, 0]
date_series = pd.Series(date_column.values, name="")  # lowercase

# Group columns by var level by walking left to right
current_var = None
current_cols_idx = []
var_to_columns = {}

for col_idx in range(1, len(var_levels_row)):  # skip date column
    var = var_levels_row[col_idx]
    if var != current_var:
        if current_var is not None:
            var_to_columns[current_var] = current_cols_idx
        current_var = var
        current_cols_idx = [col_idx]
    else:
        current_cols_idx.append(col_idx)

# Add final group
if current_var is not None:
    var_to_columns[current_var] = current_cols_idx

# Save each group to a .csv file
for var_level, indices in var_to_columns.items():
    # Extract just the numeric part (e.g., 'VaR_0.010' → '10')
    short_var = var_level.replace("VaR_", "")
    cleaned_var = short_var.replace(".", "")  # remove the dot entirely

    # Extract and label columns
    stock_names = stock_names_row[indices].values
    var_data = data[indices].copy()
    var_data.columns = stock_names

    # Combine with date
    final_df = pd.concat([date_series.reset_index(drop=True), var_data.reset_index(drop=True)], axis=1)
    final_df.columns = ["date"] + list(stock_names)

    # Save to CSV
    output_path = os.path.join(output_folder, f"VaR_{cleaned_var}.csv")
    final_df.to_csv(output_path, index=False)
