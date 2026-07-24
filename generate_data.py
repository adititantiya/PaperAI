import pandas as pd
import numpy as np

np.random.seed(42)

rows = 5000

data = {
    "stock_flow": np.random.normal(100, 10, rows),
    "filler_flow": np.random.normal(20, 3, rows),
    "steam_pressure": np.random.normal(55, 8, rows),
    "machine_speed": np.random.normal(900, 60, rows),
    "moisture": np.random.normal(5, 1, rows),
    "ash": np.random.normal(2, 0.4, rows),
    "basis_weight": np.random.normal(80, 5, rows),
    "target_basis_weight": np.random.choice(
        [70, 75, 80, 85, 90, 95],
        rows
    )
}

df = pd.DataFrame(data)

# Percentage deviation
df["bw_deviation"] = (
    abs(df["basis_weight"] - df["target_basis_weight"])
    / df["target_basis_weight"]
) * 100

# More realistic off-spec rule
conditions = (
    (df["bw_deviation"] > 2.5) |
    (df["steam_pressure"] > 65) |
    (df["machine_speed"] > 980) |
    (df["moisture"] > 6.5)
)

df["off_spec"] = conditions.astype(int)

df.to_csv("data/historical_data.csv", index=False)

print(df.head())
print("\nDataset created successfully!")