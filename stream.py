import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import numpy as np
from clusteval import clusteval

st.title("Airline Hierarchical Clustering App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data", df.head())

    # Selecting relevant columns (same as your code)
    df1 = df[["Operating Airline", "GEO Region", "Passenger Count"]]
    
    airline_count = df1["Operating Airline"].value_counts()
    passenger_count = df1.groupby("Operating Airline").sum()["Passenger Count"]

    df2 = pd.concat([airline_count, passenger_count], axis=1)
    df2.columns = ['count', 'Passenger Count']

    # Show scatter plot and identify outliers
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df2['count'], df2['Passenger Count'])
    for i, txt in enumerate(df2.index):
        ax.annotate(txt, (df2['count'][i], df2['Passenger Count'][i]), fontsize=8)
    ax.set_xlabel("Flights held")
    ax.set_ylabel("Passengers")
    st.pyplot(fig)

    # Drop outliers
    outliers = st.text_input("Enter airlines to exclude (comma separated)", 
                             "United Airlines, United Airlines - Pre 07/01/2013")
    outliers = [x.strip() for x in outliers.split(",") if x.strip() != '']
    df3 = df2.drop(index=outliers, errors='ignore')

    st.write("### Data after removing outliers", df3)
    
    # Number of clusters input
    n_clusters = st.number_input(
        "Select number of clusters",
        min_value=2,
        max_value=10,
        value=4,
        step=1
    )

    # Hierarchical Clustering
    Z = linkage(df3, method="ward")

    # Fit clustering model
    hc = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage='ward'
    )

    labels = hc.fit_predict(df3)
    df3['cluster'] = labels

    # Dendrogram
    fig2, ax2 = plt.subplots(figsize=(16, 8))

    dendrogram(Z, ax=ax2)

    # Find cutting height that produces the selected number of clusters
    cut_height = (Z[-n_clusters, 2] + Z[-n_clusters + 1, 2]) / 2

    ax2.axhline(
        y=cut_height,
        color='red',
        linestyle='--',
        label=f'{n_clusters} Clusters'
    )

    ax2.set_title('Hierarchical Clustering Dendrogram')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Euclidean distance')
    ax2.legend()

    st.pyplot(fig2)

    st.write("### Cluster assignment", df3)

    # Clustering metrics
    sil_score = metrics.silhouette_score(df3.drop(columns=['cluster']), labels)
    ch_score = metrics.calinski_harabasz_score(df3.drop(columns=['cluster']), labels)
    db_score = metrics.davies_bouldin_score(df3.drop(columns=['cluster']), labels)

    st.write(f"**Silhouette Score:** {sil_score:.3f}")
    st.write(f"**Calinski Harabasz Score:** {ch_score:.3f}")
    st.write(f"**Davies-Bouldin Score:** {db_score:.3f}")

    # Optional: Cluster evaluation plot (takes time)
    if st.button("Run Clusteval Evaluation"):
        ce = clusteval(evaluate='silhouette')
        ce.fit(np.array(df3.drop(columns=['cluster'])))
        ce.plot()
        st.pyplot(plt.gcf())
