import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from clusteval import clusteval
import os

# Set project directory
project_root = os.getcwd()
# Ensure image directory exists
#image_dir = os.path.join(project_root, "static", "images")
#os.makedirs(image_dir, exist_ok=True)

# Load data
df = pd.read_csv(r"C:\Users\Hp\Desktop\clustering_Proj\AirTraffic_Passenger_Statistics.csv")
df1 = df[["Operating Airline", "GEO Region", "Passenger Count"]]
airline_count = df1["Operating Airline"].value_counts()
passenger_count = df1.groupby("Operating Airline").sum()["Passenger Count"]

# Combine
df2 = pd.concat([airline_count, passenger_count], axis=1)
df2.columns = ["Flights held", "Passenger Count"]
df2.drop(["United Airlines", "United Airlines - Pre 07/01/2013"], inplace=True)

# Scale
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df2)

# Dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage(scaled_df, method="ward"))
plt.title("Hierarchical Clustering Dendrogram")
#dendrogram_path = os.path.join(image_dir, "dendrogram.png")
#plt.savefig(dendrogram_path)
plt.show()
plt.close()

# Agglomerative clustering
hc1 = AgglomerativeClustering(n_clusters=4, linkage='ward')
clusters = hc1.fit_predict(scaled_df)
df2['cluster'] = clusters

# Scatter plot
plt.figure(figsize=(10, 6))
for cluster in df2['cluster'].unique():
    subset = df2[df2['cluster'] == cluster]
    plt.scatter(subset['Flights held'], subset['Passenger Count'], label=f"Cluster {cluster}")
plt.xlabel("Flights Held")
plt.ylabel("Passenger Count")
plt.title("Airlines Clustering")
plt.legend()
#scatter_path = os.path.join(image_dir, "cluster_scatter.png")
#plt.savefig(scatter_path)
plt.show()
plt.close()

# Clusteval
ce = clusteval(evaluate='silhouette')
ce.fit(scaled_df)
#clusteval_plot_path = os.path.join(image_dir, 'clusteval.png')
fig_tuple = ce.plot()

# ce.plot() returns a tuple (fig, ax), so extract fig before saving
fig = fig_tuple[0]
#fig.savefig(clusteval_plot_path)

# Metrics
sil_score = metrics.silhouette_score(scaled_df, clusters)
ch_score = metrics.calinski_harabasz_score(scaled_df, clusters)
db_score = metrics.davies_bouldin_score(scaled_df, clusters)

# --- Save Outputs ---
output_csv = os.path.join(project_root, 'cluster_output.csv')
metrics_txt = os.path.join(project_root, 'metrics.txt')

# Save DataFrame
df2.to_csv(output_csv, index=True)

# Save metrics
with open(metrics_txt, "w") as f:
    f.write(f"Silhouette Score: {sil_score:.2f}\n")
    f.write(f"Calinski Harabasz Score: {ch_score:.2f}\n")
    f.write(f"Davies-Bouldin Score: {db_score:.2f}\n")

print(f"\n✅ Files saved to:", flush=True)
print(f"   CSV: {output_csv}", flush=True)
print(f"   TXT: {metrics_txt}", flush=True)
