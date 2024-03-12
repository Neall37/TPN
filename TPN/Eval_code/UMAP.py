from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import h5py
import plotly.graph_objs as go
import umap
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

h5_file_path = "Verticillium_eval_esm2_t12_35M_UR50D.h5"
labels = np.concatenate([np.ones(100), np.full(100, 2),
                         np.full(100, 3)])
with h5py.File(h5_file_path, 'a') as h5file:
    # Create a new dataset for the labels, np.full(10, 3)
    if 'labels' in h5file:
        del h5file['labels']
    h5file.create_dataset('labels', data=labels)
h5 = h5py.File(h5_file_path, 'r')

data = h5['sequences']
data = np.array(data)

batches = []
for i in range(data.shape[0]):
    batch = data[i].transpose(1, 0).reshape(-1, 480)
    batches.append(batch)

ordered_data = np.concatenate(batches, axis=0)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(ordered_data)

reducer = umap.UMAP(n_neighbors=10, min_dist=0.5, n_components=2, random_state=42)
# Fit the model to your data and transform it to a lower-dimensional space
embedding = reducer.fit_transform(data_scaled)

# Set the size of the plot
mpl.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(12, 10))
unique_labels = np.unique(labels)
markers = ['o', 's', '^', 'v', '>', '<', 'p', '*', '+', 'x']
colors = ['#00A087CC', '#DC0000CC', '#3C5488CC']
category_names = ['XP_009650171', 'YP_010799217', 'BDC60758']
# Ensure we have enough markers for the categories
if len(unique_labels) > len(markers):
    raise ValueError("Not enough markers defined for the number of categories")

# Plot each category with its own marker
for i, label in enumerate(unique_labels):
    # Filter data points belonging to the current category
    idx = labels == label
    plt.scatter(embedding[idx, 0], embedding[idx, 1], marker=markers[i], c=colors[i], s=45, label=category_names[i])


plt.gca().set_aspect('equal', 'datalim')
#plt.title('UMAP for homologous proteins \n esm2_t12_35M_UR50D', fontsize=27)
plt.xlabel('UMAP 1', fontsize=26)
plt.ylabel('UMAP 2', fontsize=26)
#plt.title('ESM-2 3B parameters', fontsize=27)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(fontsize=25)

# Show the plot without a colorbar
plt.show()


# Calculate the pairwise correlation matrix
correlation_matrix = np.corrcoef(data_scaled)
# Create a color palette for the 3 groups
labels = labels.astype(np.int32)
group_palette = ['#00A087CC', '#DC0000CC', '#3C5488CC']

# Map group identifiers to colors
group_colors = [group_palette[g-1] for g in labels]

plt.figure(figsize=(13, 10))

# Create a grid for the subplots
grid_spec = plt.GridSpec(1, 2, width_ratios=[0.1, 4.8], wspace=0.1)

# Add a subplot for group color bar
group_ax = plt.subplot(grid_spec[0])
sns.heatmap(np.column_stack([labels, labels]),
            cmap=group_palette,
            cbar=False,
            yticklabels=False,
            xticklabels=False,
            ax=group_ax)

# Add text labels for each group color bar
unique_labels = np.unique(labels)
# Calculate the center of each color strip
color_bar_centers = (np.arange(len(unique_labels)) + 0.5) / len(unique_labels)
for y, (label, center) in enumerate(zip(unique_labels, color_bar_centers)):
    # Adjust the x-position, alignment, and font size as needed
    print(category_names[y])
    print(center)
    #group_ax.text(-0.1, 1-center, f'{category_names[y]}', va='center', ha='right', fontsize=20, transform=group_ax.transAxes)

heatmap_ax = plt.subplot(grid_spec[1])
colors = ['#5E719B','white', '#DE2E2E']  # Define the colors at minimum and maximum "coolwarm"
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
heatmap = sns.heatmap(correlation_matrix, annot=False, cmap=cmap, cbar=True, ax=heatmap_ax)
heatmap_ax.set_xticks([])  # Remove x-axis ticks
heatmap_ax.set_yticks([])  # Remove y-axis ticks
# Adjust colorbar tick label size
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=27)  # Adjust to your desired size

# Set the title and labels for the heatmap
# plt.title('ESM-2 35M parameters', fontsize=27)
plt.xlabel('Proteins', fontsize=27)
plt.show()


