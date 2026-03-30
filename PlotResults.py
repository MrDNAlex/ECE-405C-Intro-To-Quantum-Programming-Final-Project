import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#
# Ryan Becze
# 20958526
# Code to plot the results shown in the presentation from the results CSV
#

# 1. List knowns (File Loading)
# Load the single combined dataset directly
df_runs = pd.read_csv('Results/CompressionBenchmarkResultsInneficientBest.csv')

# Drop any empty columns caused by trailing commas in the CSV
df_runs = df_runs.loc[:, ~df_runs.columns.str.contains('^Unnamed')]

# 2. Process Data
# Group by photo name and quality parameter to ensure data is clean and averaged
df_agg = df_runs.groupby(['Photo Name', 'Quality Parameter']).mean(numeric_only=True).reset_index()

# Load the Time vs Blocks data (Note: this is using old data when all color channels were processed at the same time)
# Any new data produced by testDCTvsQDCT only measures the Luminance Channel
df_time = pd.read_csv('Results/DCTvsQDCT.csv')
df_time = df_time[df_time['Name'] != 'Blue Marble'] 
df_time = df_time.sort_values('NumBlocks')

# Helper function to style graphs uniformly
def apply_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=8)

# ---------------------------------------------------------
# Graph 1: Compressed Size Ratio vs Quality Level
# ---------------------------------------------------------
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for photo in df_agg['Photo Name'].unique():
    subset = df_agg[df_agg['Photo Name'] == photo]
    # Standard DCT (JPEG Compression column)
    ax1.plot(subset['Quality Parameter'], subset['JPEG Compression'], marker='o', label=photo)
    # QDCT (QFT Compression column)
    ax2.plot(subset['Quality Parameter'], subset['QFT Compression'], marker='s', label=photo)

apply_style(ax1, 'Standard DCT: Compression Ratio vs Quality', 'Quality Level', 'Compression Ratio')
apply_style(ax2, 'QDCT: Compression Ratio vs Quality', 'Quality Level', 'Compression Ratio')
fig1.tight_layout()

# ---------------------------------------------------------
# Graph 2: Computation Time vs Number of Blocks
# ---------------------------------------------------------
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

# --- Standard DCT ---
x_dct = df_time['NumBlocks']
y_dct = df_time['DCT Time']

ax3.plot(x_dct, y_dct, marker='o', linestyle='none', color='tab:blue', label='Standard DCT Time')

# Calculate trendline and R^2 for DCT
z_dct = np.polyfit(x_dct, y_dct, 1)
p_dct = np.poly1d(z_dct)
yhat_dct = p_dct(x_dct)                           # Predicted y values
ybar_dct = np.mean(y_dct)                         # Average y value
ssreg_dct = np.sum((yhat_dct - ybar_dct)**2)      # Regression Sum of Squares
sstot_dct = np.sum((y_dct - ybar_dct)**2)         # Total Sum of Squares
r2_dct = ssreg_dct / sstot_dct                    # R-squared value

# Format the equation and R^2 label
label_dct = f'Trendline: y = {z_dct[0]:.2e}x + {z_dct[1]:.4f}\n$R^2$ = {r2_dct:.4f}'
ax3.plot(x_dct, yhat_dct, color='tab:blue', linestyle='--', alpha=0.6, label=label_dct)

# --- QDCT ---
x_qdct = df_time['NumBlocks']
y_qdct = df_time['QDCT Time']

ax4.plot(x_qdct, y_qdct, marker='s', linestyle='none', color='tab:orange', label='QDCT Time')

# Calculatetrendline and R^2 for QDCT
z_qdct = np.polyfit(x_qdct, y_qdct, 1)
p_qdct = np.poly1d(z_qdct)
yhat_qdct = p_qdct(x_qdct)
ybar_qdct = np.mean(y_qdct)
ssreg_qdct = np.sum((yhat_qdct - ybar_qdct)**2)
sstot_qdct = np.sum((y_qdct - ybar_qdct)**2)
r2_qdct = ssreg_qdct / sstot_qdct

# Format the equation and R^2 label
label_qdct = f'Trendline: y = {z_qdct[0]:.2e}x + {z_qdct[1]:.4f}\n$R^2$ = {r2_qdct:.4f}'
ax4.plot(x_qdct, yhat_qdct, color='tab:orange', linestyle='--', alpha=0.6, label=label_qdct)

# Annotate points with the image names
for index, row in df_time.iterrows():
    ax3.annotate(row['Name'], (row['NumBlocks'], row['DCT Time']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    ax4.annotate(row['Name'], (row['NumBlocks'], row['QDCT Time']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

apply_style(ax3, 'Standard DCT: Time vs Blocks', 'Number of Blocks', 'Computation Time (s)')
apply_style(ax4, 'QDCT: Time vs Blocks', 'Number of Blocks', 'Computation Time (s)')

fig2.tight_layout()

# ---------------------------------------------------------
# Graph 3: Computation Time vs Final Compressed Size
# ---------------------------------------------------------
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 7))

# Filter out Blue Marble to keep the scale readable
df_agg_filtered = df_agg[df_agg['Photo Name'] != 'Blue Marble']
photos = df_agg_filtered['Photo Name'].unique()

# Create a list of different shapes (markers)
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '<']

for i, photo in enumerate(photos):
    subset = df_agg_filtered[df_agg_filtered['Photo Name'] == photo].sort_values('Quality Parameter')
    
    color = f'C{i}' 
    marker_shape = markers[i % len(markers)]
    
    # --- Standard DCT Subplot ---
    dct_x, dct_y = subset['CV2 Size (KB)'], subset['Time DCT (s)']
    ax5.plot(dct_x, dct_y, marker=marker_shape, linestyle='none', color=color, label=photo)
    
    # --- QFT Subplot ---
    qft_x, qft_y = subset['QJPEG Size (KB)'], subset['Time QFT (s)']
    ax6.plot(qft_x, qft_y, marker=marker_shape, linestyle='none', color=color, label=photo)

# Style both subplots independently (Updated labels for log base 2 scale)
apply_style(ax5, 'Standard DCT Time vs Compressed Size', 'CV2 Size (KB) (Log10 Scale)', 'Computation Time (s)')
apply_style(ax6, 'QFT Time vs Compressed Size', 'QJPEG Size (KB) (Log10 Scale)', 'Computation Time (s)')

# Set the x-axis to a logarithmic scale with base 2
ax5.set_xscale('log', base=10)
ax6.set_xscale('log', base=10)

fig3.tight_layout()

plt.show()