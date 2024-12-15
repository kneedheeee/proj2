import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import zscore
import requests


# Function to safely load datasets with encoding fallback
def load_data_with_fallback(filepath):
    """
    Attempts to load a dataset with UTF-8 encoding, falling back to Latin-1 if UTF-8 fails.
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
    corr_matrix = data.select_dtypes(exclude="object").corr()  # Correlation matrix for numeric columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Correlation Coefficient"}
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.close()


# Perform PCA for dimensionality reduction
def apply_pca(data, components=2):
    """
    Performs PCA on numeric columns of the dataset, reducing dimensions and returning results and variance explained.
    """
    numeric_data = data.select_dtypes(include=[np.number]).dropna()
    pca_model = PCA(n_components=components)
    reduced_data = pca_model.fit_transform(numeric_data)
    variance_explained = pca_model.explained_variance_ratio_
    return reduced_data, variance_explained


# Cluster data using KMeans
def perform_kmeans_clustering(data, plot_filename, clusters_count=3):
    """
    Performs KMeans clustering on the dataset and generates a scatter plot of clusters.
    """
    # Clean data for clustering
    numeric_data = data.select_dtypes(include=[np.number]).dropna()

    if numeric_data.empty:
        print("No numeric data available for clustering.")
        return

    # Initialize and fit KMeans model
    kmeans_model = KMeans(n_clusters=clusters_count, random_state=42)
    cluster_labels = kmeans_model.fit_predict(numeric_data)

    # Add cluster labels to the dataset
    data["Cluster"] = cluster_labels

    # Scatter plot of clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(
        numeric_data.iloc[:, 0],
        numeric_data.iloc[:, 1],
        c=cluster_labels,
        cmap="viridis",
        alpha=0.7
    )
    plt.title(f"KMeans Clustering (k={clusters_count})")
    plt.xlabel(numeric_data.columns[0])
    plt.ylabel(numeric_data.columns[1])
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    print(f"Clustering complete. Plot saved to {plot_filename}")


# Query the AI model for insights
def query_ai_model(prompt, token_limit=300):
    """
    Sends a prompt to the AI model and returns the response.
    """
    os.environ["AIPROXY_TOKEN"] = "your_token_here"
    headers = {
        "Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}",
        "Content-Type": "application/json"
    }

    request_payload = {
        "model": "gpt-4o-mini",
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


# Clean data and detect outliers
def clean_data_and_find_outliers(dataset):
    """
    Cleans data by removing rows with NaN values and detects outliers using z-scores.
    """
    cleaned_data = dataset.dropna()
    z_scores = np.abs(zscore(cleaned_data.select_dtypes(include=[np.number])))
    outlier_mask = (z_scores > 3).any(axis=1)
    cleaned_data = cleaned_data[~outlier_mask]
    return cleaned_data, outlier_mask


# Main workflow
def run_analysis(file_path):
    """
    The main function orchestrating data analysis and reporting.
    """
    dataset = load_data_with_fallback(file_path)

    # Clean and analyze data
    cleaned_data, outliers = clean_data_and_find_outliers(dataset)
    print(f"Outliers detected: {np.sum(outliers)}")

    # Generate visualizations
    create_correlation_heatmap(cleaned_data, "correlation_heatmap.png")

    # PCA and clustering
    pca_results, variance = apply_pca(cleaned_data)
    print(f"Explained variance: {variance}")
    perform_kmeans_clustering(cleaned_data, "cluster_plot.png")

    # AI insights
    ai_prompt = f"Analyze dataset columns and trends:\n{cleaned_data.dtypes}"
    ai_insights = query_ai_model(ai_prompt)
    print("AI Insights:", ai_insights)

    # Generate Markdown report
    with open("README.md", "w") as f:
        f.write("# Data Analysis Report\n\n")
        f.write(f"Dataset contains {len(dataset)} rows and {len(dataset.columns)} columns.\n")
        f.write(f"Outliers detected: {np.sum(outliers)}\n")
        f.write("## PCA Explained Variance\n")
        f.write(f"{variance}\n")
        f.write("## Insights\n")
        f.write(ai_insights)
        f.write("\n\n## Visualizations\n")
        f.write("![Correlation Heatmap](correlation_heatmap.png)\n")
        f.write("![Cluster Plot](cluster_plot.png)\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset_path>")
        sys.exit(1)
    run_analysis(sys.argv[1])
