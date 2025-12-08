import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style for "Premium" look
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (12, 8)
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#34495e"]
sns.set_palette(sns.color_palette(colors))

# Paths
DATA_DIR = '/Users/ricardoramos/Projects/AC/project_data/initial_data'
PLAYERS_TEAMS_PATH = os.path.join(DATA_DIR, 'players_teams.csv')
PLAYERS_PATH = os.path.join(DATA_DIR, 'players.csv')
AWARDS_PATH = os.path.join(DATA_DIR, 'awards_players.csv')

# Load data
print("Loading data...")
pt = pd.read_csv(PLAYERS_TEAMS_PATH)
players = pd.read_csv(PLAYERS_PATH)
awards = pd.read_csv(AWARDS_PATH)

# --- Preprocessing ---

# 1. Merge Position data
# players.csv has 'bioID' which corresponds to 'playerID'
players = players.rename(columns={'bioID': 'playerID'})
df = pt.merge(players[['playerID', 'pos']], on='playerID', how='left')

# 2. Merge Awards data
# We only care if they won ANY award in that year for this visualization
awards['has_award'] = 1
# Drop duplicates if a player won multiple awards in one year
awards_unique = awards[['playerID', 'year', 'has_award']].drop_duplicates()
df = df.merge(awards_unique, on=['playerID', 'year'], how='left')
df['has_award'] = df['has_award'].fillna(0).astype(int)

# 3. Calculate Efficiency (EFF)
# EFF = (PTS + REB + AST + STL + BLK - (FGA - FGM) - (FTA - FTM) - TO)
df['Missed_FG'] = df['fgAttempted'] - df['fgMade']
df['Missed_FT'] = df['ftAttempted'] - df['ftMade']
df['EFF'] = (df['points'] + df['rebounds'] + df['assists'] + df['steals'] + df['blocks'] 
             - df['Missed_FG'] - df['Missed_FT'] - df['turnovers'])

# Normalize by Games Played for fair comparison (Per Game Stats)
df = df[df['GP'] > 0] # Avoid division by zero
df['PPG'] = df['points'] / df['GP']
df['RPG'] = df['rebounds'] / df['GP']
df['APG'] = df['assists'] / df['GP']
df['EFF_PG'] = df['EFF'] / df['GP']

# Filter for meaningful analysis (e.g., players with > 10 games)
df_filtered = df[df['GP'] > 10]

print("Data preprocessed. Generating graphs...")

# --- Graph 1: Correlation Heatmap (Multicollinearity Check) ---
plt.figure(figsize=(10, 8))
cols_to_corr = ['PPG', 'RPG', 'APG', 'steals', 'blocks', 'turnovers', 'minutes', 'EFF_PG']
corr = df_filtered[cols_to_corr].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=.5)
plt.title('Correlation Matrix of Key Performance Metrics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png', dpi=300)
print("Generated eda_correlation_heatmap.png")

# --- Graph 2: Award Winners vs Non-Winners (Feature Importance) ---
# Boxplot of Efficiency per Game split by Award Status
plt.figure(figsize=(8, 6))
sns.boxplot(x='has_award', y='EFF_PG', data=df_filtered, palette=['#95a5a6', '#f1c40f'])
plt.xticks([0, 1], ['Non-Winners', 'Award Winners'], fontsize=12)
plt.xlabel('Award Status', fontsize=12)
plt.ylabel('Efficiency Rating Per Game', fontsize=12)
plt.title('Impact of Efficiency on Award Success', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_award_separation.png', dpi=300)
print("Generated eda_award_separation.png")

# --- Graph 3: Positional Roles (Cluster Analysis) ---
# Scatter plot of Assists vs Rebounds colored by Position
# This helps verify if positions have distinct statistical profiles
plt.figure(figsize=(10, 7))
# Clean up positions (keep main ones if messy)
main_positions = ['G', 'F', 'C', 'G-F', 'F-C'] # Adjust based on actual data
df_pos = df_filtered[df_filtered['pos'].isin(main_positions)]

sns.scatterplot(data=df_pos, x='RPG', y='APG', hue='pos', alpha=0.6, s=60)
plt.title('Positional Clustering: Rebounds vs Assists', fontsize=16, fontweight='bold')
plt.xlabel('Rebounds Per Game', fontsize=12)
plt.ylabel('Assists Per Game', fontsize=12)
plt.legend(title='Position')
plt.tight_layout()
plt.savefig('eda_position_clusters.png', dpi=300)
print("Generated eda_position_clusters.png")

# --- Graph 4: Scoring Distribution (Target Variable Analysis) ---
# Histogram of PPG
plt.figure(figsize=(10, 6))
sns.histplot(df_filtered['PPG'], bins=40, kde=True, color='#3498db', line_kws={'linewidth': 3})
plt.axvline(df_filtered['PPG'].mean(), color='red', linestyle='--', label=f"Mean: {df_filtered['PPG'].mean():.1f}")
plt.title('Distribution of Points Per Game (PPG)', fontsize=16, fontweight='bold')
plt.xlabel('Points Per Game', fontsize=12)
plt.ylabel('Count of Players', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('eda_ppg_distribution.png', dpi=300)
print("Generated eda_ppg_distribution.png")

