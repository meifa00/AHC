import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer
import os
import kaggle

# Streamlit app
st.title("Soccer Player Clustering Dashboard")

# Data input
st.sidebar.header("Data Input")
data_source = st.sidebar.radio("Choose Data Source", ["Upload CSV", "Fetch from Kaggle"])

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        player_with_attributes_df = pd.read_csv(uploaded_file)
else:
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('hugomathien/soccer', path='/tmp', unzip=True)
    player_with_attributes_df = pd.read_csv('/tmp/player_att_merged.csv')

# Preprocessing
if 'player_with_attributes_df' in locals():
    st.header("Data Preprocessing")
    
    # Calculate age
    player_with_attributes_df['birthday'] = pd.to_datetime(player_with_attributes_df['birthday'])
    def calculate_age(birthday):
        today = datetime.now()
        age = today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))
        return age
    player_with_attributes_df['age'] = player_with_attributes_df['birthday'].apply(calculate_age)

    player_with_attributes_df['date'] = pd.to_datetime(player_with_attributes_df['date'])
    player_with_attributes_df_2016 = player_with_attributes_df[player_with_attributes_df['date'].dt.year == 2016]
    player_with_attributes_df_2016 = player_with_attributes_df_2016.sort_values(by='date', ascending=False)
    player_with_attributes_df_2016 = player_with_attributes_df_2016.drop_duplicates(subset=['player_fifa_api_id', 'player_api_id'], keep='first')

    # Normalization
    numerical_features = player_with_attributes_df_2016.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    player_with_attributes_df_2016[numerical_features] = scaler.fit_transform(player_with_attributes_df_2016[numerical_features])

    # Encode categorical variables
    player_with_attributes_df_2016 = pd.get_dummies(player_with_attributes_df_2016, columns=['preferred_foot', 'attacking_work_rate', 'defensive_work_rate'])

    # Feature selection using SelectKBest
    X = player_with_attributes_df_2016.drop(columns=['id', 'player_api_id', 'player_name', 'player_fifa_api_id', 'birthday', 'date', 'overall_rating'])
    y = player_with_attributes_df_2016['overall_rating']
    selector = SelectKBest(score_func=f_classif, k=10)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    st.write("Selected features using SelectKBest:", selected_features)

    # PCA implementation
    pca = PCA()
    pca.fit(player_with_attributes_df_2016[selected_features])

    # Cumulative variance ratio
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_) * 100
    st.write(cumulative_variance_ratio)

    # Plot cumulative explained variance
    plt.figure(figsize=[10, 5])
    plt.title('Cumulative Explained Variance explained by component')
    plt.ylabel('Cumulative Explained variance (%)')
    plt.xlabel('Principal components')
    plt.plot(cumulative_variance_ratio)
    st.pyplot(plt)

    # How many PCs explain 95% of the variance?
    k = np.argmax(cumulative_variance_ratio > 95)
    st.write("Number of components explaining 95% variance: " + str(k))

    # PCA transformation
    pca = PCA(n_components=k)
    player_with_attributes_df_2016_pca = pca.fit_transform(player_with_attributes_df_2016[selected_features])
    st.write('Principal Components:')
    st.write(player_with_attributes_df_2016_pca)

    # AHC clustering
    st.header("Agglomerative Hierarchical Clustering")
    n_clusters = st.slider("Number of Clusters", 2, 10, 2)
    ahc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    clusters = ahc.fit_predict(player_with_attributes_df_2016_pca)

    # Visualize dendrogram
    linked = linkage(player_with_attributes_df_2016_pca, 'ward')
    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram for Agglomerative Hierarchical Clustering")
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    st.pyplot(plt)

    # Evaluation metrics
    silhouette_avg = silhouette_score(player_with_attributes_df_2016_pca, clusters)
    calinski_harabasz_avg = calinski_harabasz_score(player_with_attributes_df_2016_pca, clusters)
    davies_bouldin_avg = davies_bouldin_score(player_with_attributes_df_2016_pca, clusters)

    st.write(f'Silhouette Score: {silhouette_avg}')
    st.write(f'Calinski-Harabasz Index: {calinski_harabasz_avg}')
    st.write(f'Davies-Bouldin Index: {davies_bouldin_avg}')

    # Cross-validation
    silhouette_scorer = make_scorer(silhouette_score, greater_is_better=True)
    scores = cross_val_score(ahc, player_with_attributes_df_2016_pca, cv=5, scoring=silhouette_scorer)
    st.write(f'Cross-Validation Silhouette Scores: {scores}')
    st.write(f'Mean Cross-Validation Silhouette Score: {scores.mean()}')

    # Holdout Validation
    X_train, X_test = train_test_split(player_with_attributes_df_2016_pca, test_size=0.2, random_state=42)
    ahc.fit(X_train)
    clusters_test = ahc.fit_predict(X_test)
    silhouette_avg_test = silhouette_score(X_test, clusters_test)
    st.write(f'Test Silhouette Score: {silhouette_avg_test}')

    # Plot clusters
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(player_with_attributes_df_2016_pca[:, 0], player_with_attributes_df_2016_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.6, edgecolor='k')
    unique_labels = np.unique(clusters)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / (len(unique_labels) - 1)), markersize=10, label=f'Cluster {i}') for i in unique_labels]
    plt.legend(handles=handles)
    plt.title('Agglomerative Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    st.pyplot(plt)
