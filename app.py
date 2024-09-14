import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

st.title("European Soccer Player Analysis Dashboard")

# Step 1: Upload CSV File
st.subheader("Upload the Player Attributes CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    player_with_attributes_df = pd.read_csv(uploaded_file)

    # Step 2: Preprocess Data
    st.subheader("Data Preprocessing")
    
    # Calculate age
    player_with_attributes_df['birthday'] = pd.to_datetime(player_with_attributes_df['birthday'])

    def calculate_age(birthday):
        today = datetime.now()
        age = today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))
        return age

    player_with_attributes_df['age'] = player_with_attributes_df['birthday'].apply(calculate_age)

    # Select data for 2016
    player_with_attributes_df['date'] = pd.to_datetime(player_with_attributes_df['date'])
    player_with_attributes_df_2016 = player_with_attributes_df[player_with_attributes_df['date'].dt.year == 2016]
    player_with_attributes_df_2016 = player_with_attributes_df_2016.sort_values(by='date', ascending=False)
    player_with_attributes_df_2016 = player_with_attributes_df_2016.drop_duplicates(subset=['player_fifa_api_id', 'player_api_id'], keep='first')

    # Normalize numerical features
    st.write("Standardizing Numerical Features...")
    numerical_features = player_with_attributes_df_2016.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    player_with_attributes_df_2016[numerical_features] = scaler.fit_transform(player_with_attributes_df_2016[numerical_features])

    # Encode categorical variables
    player_with_attributes_df_2016 = pd.get_dummies(player_with_attributes_df_2016, columns=['preferred_foot', 'attacking_work_rate', 'defensive_work_rate'])

    st.write("Data Sample After Preprocessing:")
    st.dataframe(player_with_attributes_df_2016.head())

    # Step 3: Feature Selection (PCA)
    st.subheader("PCA and Feature Selection")
    
    # Feature selection
    X = player_with_attributes_df_2016.drop(columns=['id', 'player_api_id', 'player_name', 'player_fifa_api_id', 'birthday', 'date', 'overall_rating'])
    y = player_with_attributes_df_2016['overall_rating']

    # PCA Implementation
    pca = PCA()
    X_pca = pca.fit_transform(X)

    # Plot cumulative variance
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_) * 100
    plt.figure(figsize=(10, 5))
    plt.title('Cumulative Explained Variance')
    plt.ylabel('Cumulative Explained variance (%)')
    plt.xlabel('Principal Components')
    plt.plot(cumulative_variance_ratio)
    st.pyplot(plt)

    # Allow user to select how many components to use
    k = st.slider('Select the number of principal components to keep (to explain at least 95% variance)', 1, len(cumulative_variance_ratio), int(np.argmax(cumulative_variance_ratio > 95) + 1))
    st.write(f"Selected {k} components.")

    # Apply PCA with selected components
    pca = PCA(n_components=k)
    player_with_attributes_df_2016_pca = pca.fit_transform(X)

    # Step 4: Clustering (Agglomerative Hierarchical Clustering)
    st.subheader("Agglomerative Hierarchical Clustering")

    # Perform clustering
    ahc = AgglomerativeClustering(n_clusters=2, linkage='ward')
    clusters = ahc.fit_predict(player_with_attributes_df_2016_pca)

    # Visualize dendrogram
    st.write("Dendrogram")
    linked = linkage(player_with_attributes_df_2016_pca, 'ward')
    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram for AHC")
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    st.pyplot(plt)

    # Cluster evaluation metrics
    st.write("Cluster Evaluation Metrics")
    silhouette_avg = silhouette_score(player_with_attributes_df_2016_pca, clusters)
    calinski_harabasz_avg = calinski_harabasz_score(player_with_attributes_df_2016_pca, clusters)
    davies_bouldin_avg = davies_bouldin_score(player_with_attributes_df_2016_pca, clusters)

    st.write(f"Silhouette Score: {silhouette_avg:.2f}")
    st.write(f"Calinski-Harabasz Index: {calinski_harabasz_avg:.2f}")
    st.write(f"Davies-Bouldin Index: {davies_bouldin_avg:.2f}")

    # Scatter plot of clusters
    st.write("Cluster Visualization")
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(player_with_attributes_df_2016_pca[:, 0], player_with_attributes_df_2016_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.6, edgecolor='k')

    # Add cluster labels to the plot
    unique_labels = np.unique(clusters)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / (len(unique_labels) - 1)), markersize=10, label=f'Cluster {i}') for i in unique_labels]
    plt.legend(handles=handles)
    plt.title("Agglomerative Clustering (AHC)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(plt)
