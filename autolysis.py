# /// script
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "scikit-learn",
#   "numpy",
#   "scipy",
#   "requests<3",
#   "tabulate",
# ]
# ///

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import zscore
import requests

# Function to load datasets safely
def load_data(filepath):
    """Attempts to load a dataset with fallback encoding."""
    try:
        return pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding="latin-1")

# Data cleaning: Handle NaNs and detect outliers
def clean_and_analyze_data(data):
    """Cleans the dataset and performs basic exploratory analysis."""
    cleaned_data = data.dropna()
    numeric_data = cleaned_data.select_dtypes(include=[np.number])
    z_scores = np.abs(zscore(numeric_data))
    outlier_mask = (z_scores > 3).any(axis=1)
    cleaned_data = cleaned_data[~outlier_mask]
    descriptive_stats = numeric_data.describe()
    return cleaned_data, descriptive_stats, outlier_mask

# Generate visualizations
def create_visualizations(data, output_file):
    """Creates a combined heatmap and clustering visualization."""
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()

    plt.figure(figsize=(16, 8))

    # Heatmap
    plt.subplot(1, 2, 1)
    plt.title("Correlation Heatmap")
    cax = plt.matshow(corr_matrix, cmap="coolwarm", fignum=0)
    plt.colorbar(cax)

    # Clustering Visualization
    plt.subplot(1, 2, 2)
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(numeric_data)
    plt.scatter(numeric_data.iloc[:, 0], numeric_data.iloc[:, 1], c=cluster_labels, cmap="viridis")
    plt.title("KMeans Clustering")
    plt.xlabel(numeric_data.columns[0])
    plt.ylabel(numeric_data.columns[1])

    plt.savefig(output_file, dpi=300)
    plt.close()

# Perform PCA
def perform_pca(data, components=2):
    """Reduces dimensionality using PCA and returns explained variance."""
    numeric_data = data.select_dtypes(include=[np.number]).dropna()
    pca = PCA(n_components=components)
    reduced_data = pca.fit_transform(numeric_data)
    explained_variance = pca.explained_variance_ratio_
    return reduced_data, explained_variance

# Query the AI model
def query_ai_model(prompt, token_limit=300):
    """Sends a concise prompt to the AI model with cost-efficient settings."""
    os.environ["AIPROXY_TOKEN"] = "your_token_here"
    headers = {
        "Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}",
        "Content-Type": "application/json"
    }
    request_payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": token_limit,
        "detail": "low"  # Optimize cost
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

# Generate report
def generate_report(file_path, stats, variance, ai_insights, output_file):
    """Creates a Markdown report summarizing the analysis."""
    with open("README.md", "w") as f:
        f.write(f"# Data Analysis Report\n\n")
        f.write(f"**Dataset:** {file_path}\n\n")
        f.write(f"## Descriptive Statistics\n\n{stats.to_markdown()}\n\n")
        f.write(f"## PCA Explained Variance\n\n{variance}\n\n")
        f.write(f"## AI Insights\n\n{ai_insights}\n\n")
        f.write(f"## Visualizations\n\n![Analysis Visualization]({output_file})\n")

# Main function
def run_analysis(file_path):
    """Orchestrates the workflow for data analysis."""
    dataset = load_data(file_path)
    cleaned_data, stats, outliers = clean_and_analyze_data(dataset)

    print(f"Outliers removed: {outliers.sum()}")

    # Generate visualizations
    visualization_file = "analysis_visualization.png"
    create_visualizations(cleaned_data, visualization_file)

    # Perform PCA
    _, explained_variance = perform_pca(cleaned_data)

    # Query AI model with summarized column details
    ai_prompt = f"Dataset columns:\n{dataset.dtypes.to_string()}\nProvide insights on trends and anomalies."
    ai_insights = query_ai_model(ai_prompt)

    # Generate report
    generate_report(file_path, stats, explained_variance, ai_insights, visualization_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset_path>")
        sys.exit(1)
    run_analysis(sys.argv[1])
