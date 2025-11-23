import main_withComments
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.DataFrame()

for seed in range(1, 100):
    result = main_withComments.run_simulation(seed)
    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)



print("\n===== Simulation Summary Table (100 replications) =====\n")
print(df.to_string(index=False))

summary = df.describe(include='all').loc[["mean", "std", "min", "max"]]
print("\n===== Summary Statistics (mean, std, min, max) =====\n")
print(summary.to_string())

print("\n===== Key averages across replications =====")
for col in df.columns:
    if col != "seed":
        print(f"{col:35s} : {df[col].mean():.4f}")

csv_path = "simulation_summary_k=2.csv"
df.to_csv(csv_path, index=False)
print(f"\nCSV saved to: {csv_path}")

xlsx_path = "simulation_summary_k=2.xlsx"
df.to_excel(xlsx_path, index=False)
print(f"Excel saved to: {xlsx_path}")

