import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to players.csv (in the same folder as this script)
file_path = os.path.join(script_dir, "players.csv")

# Load the dataset
df = pd.read_csv(file_path)

# Drop missing values for height and weight
df = df.dropna(subset=["height", "weight"])

plt.figure(figsize=(8, 6))
positions = df["pos"].fillna("Unknown")
for pos in positions.unique():
    subset = df[positions == pos]
    plt.scatter(subset["height"], subset["weight"], alpha=0.6, edgecolor="k", label=pos)

plt.title("Height vs. Weight of Players by Position")
plt.xlabel("Height (inches)")
plt.ylabel("Weight (lbs)")
plt.legend(title="Position")
plt.grid(alpha=0.3)
plt.show()
