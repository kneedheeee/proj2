import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import zscore
import openai
import matplotlib.cm as cm
import requests


# Function to safely load datasets, with encoding adjustments
def load_data_with_fallback(filepath):
    """
    Attempts to load a dataset, first with UTF-8 encoding, then falls back to Latin-1 if UTF-8 fails.
    """
    try:
        return pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding="latin-1")


# Create a heatmap for data correlations
def create_correlation_heatmap(data, save_as):
    """
    Generates a heatmap visualizing the correlation between numerical features in the dataset.
    The output is saved as a PNG file.
    """
    corr_matrix = data.select_dtypes(exclude="object").corr()  # Get the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.savefig(save_as, dpi=300)
    plt.close()


# Execute PCA for dimensionality reduction
def apply_pca(data, components=2):
    """
    Performs PCA on the dataset, reducing its dimensions and returning the transformed data along with the variance explained.
    """
    numeric_data = data.select_dtypes(include=[np.number]).dropna()
    pca_model = PCA(n_components=components)
    reduced_data = pca_model.fit_transform(numeric_data)
    variance_explained = pca_model.explained_variance_ratio_
    return reduced_data, variance_explained


# Cluster data using KMeans
def perform_kmeans_clustering(data, plot_filename):
    """
    Performs KMeans clustering on the dataset, adds the resulting cluster labels, and generates a scatter plot.
    """
    if data.isnull().any().any():
        print("Missing values found. Please clean the dataset before clustering.")
        return

    # Select numeric columns for clustering
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    if numeric_data.empty:
        print("No numeric data found for clustering.")
        return

    # Initialize and fit the KMeans model
    clusters_count = 3  # Specify number of clusters
    kmeans_model = KMeans(n_clusters=clusters_count, random_state=42)
    cluster_labels = kmeans_model.fit_predict(numeric_data)

    # Match cluster labels with the dataset
    if len(cluster_labels) != len(data):
        print(f"Cluster count mismatch: {len(cluster_labels)} vs. dataset rows ({len(data)})")
        if len(cluster_labels) < len(data):
            cluster_labels = list(cluster_labels) + [None] * (len(data) - len(cluster_labels))
        else:
            cluster_labels = cluster_labels[:len(data)]

    data["Cluster"] = cluster_labels

    # Plot the clusters (adjust based on data's dimensionality)
    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=cluster_labels, cmap='viridis')
    plt.title('Clustering Visualization')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.colorbar(label='Cluster')
    plt.savefig(plot_filename)
    plt.show()

    print(f"Clustering complete. Plot saved to {plot_filename}")


# Function to query the AI Model
def query_ai_model(prompt, token_limit=300):
    """
    Sends a prompt to the AI model for analysis via the AI Proxy service.
    """
    os.environ["AIPROXY_TOKEN"] = "your_token_here"
    headers = {
        "Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}",
        "Content-Type": "application/json"
    }

    request_payload = {
        "model": "gpt-4o-mini",  # Ensure the model is correct
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": token_limit
    }

    try:
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers=headers,
            json=request_payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error querying AI model: {str(e)}"


# Data cleaning: Handle NaNs and detect outliers via z-scores
def clean_data_and_find_outliers(dataset):
    """
    Removes rows with missing values and identifies outliers based on z-scores.
    """
    cleaned_data = dataset.dropna()  # Remove rows with NaN values
    z_scores = np.abs(zscore(cleaned_data.select_dtypes(include=[np.number])))
    outlier_mask = (z_scores > 3).all(axis=1)  # Flag rows with outliers
    cleaned_data = cleaned_data[~outlier_mask]
    
    return cleaned_data, outlier_mask


# Main control function to manage data analysis workflow
def run_analysis(file_path):
    """
    The central function that orchestrates the workflow, from loading data to generating reports.
    """
    # Load the dataset
    dataset = load_data_with_fallback(file_path)

    # Clean and analyze data
    cleaned_data, outliers = clean_data_and_find_outliers(dataset)
    print(f"Outliers detected: {np.sum(outliers)}")

    # Generate the correlation heatmap
    create_correlation_heatmap(dataset, "correlation_heatmap.png")

    # Perform PCA and display explained variance
    pca_results, variance = apply_pca(dataset)
    print(f"Explained variance: {variance}")

    # Perform clustering and generate plot
    perform_kmeans_clustering(dataset, "cluster_plot.png")

    # Generate insights from AI
    ai_prompt = f"Analyze the dataset columns and types:\n{dataset.dtypes}\nProvide insights on trends and anomalies."
    ai_insights = query_ai_model(ai_prompt)
    print("AI Insights:", ai_insights)

    # Write a report in Markdown format
    with open("README.md", "w") as f:
        f.write("# Data Analysis Report\n\n")
        f.write(f"Dataset contains {len(dataset)} rows and {len(dataset.columns)} columns.\n")
        f.write("## Insights\n")
        f.write(ai_insights)
        f.write("\n\n## Visualizations\n")
        f.write("![Correlation Heatmap](correlation_heatmap.png)\n")
        f.write("![Cluster Plot](cluster_plot.png)\n")

    # Advanced analysis via AI
    advanced_prompt = f"Describe patterns and relationships based on PCA and clustering results."
    advanced_ai_insights = query_ai_model(advanced_prompt)
    print("Advanced AI Insights:", advanced_ai_insights)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset_path>")
        sys.exit(1)
    run_analysis(sys.argv[1])
