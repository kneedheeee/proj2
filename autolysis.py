# /// script
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "scikit-learn",
#   "numpy",
#   "scipy",
#   "requests<3",
#   "tabulate",
#   "seaborn",
# ]
# ///

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import zscore
import requests

# Load and clean data
def load_and_clean_data(filepath):
    """
    Loads the dataset and performs initial cleaning by handling missing values and detecting outliers.
    """
    try:
        data = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(filepath, encoding='latin-1')

    data = data.dropna()  # Remove rows with missing values
    numeric_data = data.select_dtypes(include=[np.number])
    z_scores = np.abs(zscore(numeric_data))
    data = data[(z_scores < 3).all(axis=1)]  # Remove outliers
    return data

# Perform exploratory data analysis
def perform_eda(data):
    """
    Performs descriptive statistics and visualizations for exploratory data analysis.
    """
    desc_stats = data.describe()
    print("Descriptive Statistics:\n", desc_stats)

    numeric_data = data.select_dtypes(include=[np.number])  # Ensure only numeric data
    corr_matrix = numeric_data.corr()

    # Create combined visualization
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")

    return desc_stats, corr_matrix

# Perform clustering and create visualization
def perform_clustering_and_visualization(data, n_clusters=3):
    """
    Performs KMeans clustering, adds cluster labels to the dataset, and creates a PCA scatter plot with clusters.
    """
    numeric_data = data.select_dtypes(include=[np.number])  # Ensure only numeric data

    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(numeric_data)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(numeric_data)
    data['Cluster'] = cluster_labels

    # Combined visualization
    plt.figure(figsize=(16, 8))

    # Correlation heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")

    # PCA scatter plot with clustering
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.title("PCA Components with Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.tight_layout()
    plt.savefig("analysis_visualization.png", dpi=300)
    plt.close()

    return pca.explained_variance_ratio_

# Query AI for insights
def query_ai(prompt, token_limit=500):
    """
    Queries the AI Proxy service with the given prompt and returns the response.
    """
    # Ensure the AIPROXY_TOKEN is set correctly
    token = os.getenv("AIPROXY_TOKEN", "yout_token")  # Replace 'your_token_here' with your actual token

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": token_limit,
        "detail":"low"
    }

    try:
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            return "Error: Unauthorized. Your token might be expired or invalid. Please log in at https://aiproxy.sanand.workers.dev/ to get a new token."
        return f"HTTP error occurred: {e}"
    except requests.exceptions.RequestException as e:
        return f"Error querying AI model: {e}"

# Generate report
def generate_report(data, pca_variance, eda_insights, ai_insights):
    """
    Generates a Markdown report summarizing the analysis.
    """
    intro_prompt = "Write a detailed introduction for a data analysis report on a dataset, including objectives and methods used."
    intro = query_ai(intro_prompt)

    # Only include key statistics and insights for EDA
    eda_prompt = f"Provide an in-depth analysis of the following descriptive statistics and correlation insights:\n{eda_insights.describe().to_string()}"
    eda_explanation = query_ai(eda_prompt)

    # Focus on first few PCA components for explanation
    pca_prompt = f"Explain the significance of these PCA explained variance ratios: {pca_variance[:2]}. Why are they important?"
    pca_explanation = query_ai(pca_prompt)

    conclusion_prompt = "Write a comprehensive conclusion for a data analysis report, summarizing findings, insights, and implications for the dataset."
    conclusion = query_ai(conclusion_prompt)

    with open("README.md", "w") as f:
        f.write("# Data Analysis Report\n\n")
        f.write("## Introduction\n")
        f.write(intro + "\n\n")

        f.write("## Exploratory Data Analysis\n")
        f.write(eda_explanation + "\n\n")

        f.write("## PCA Explained Variance\n")
        f.write(pca_explanation + "\n\n")

        f.write("## AI Insights\n")
        f.write(ai_insights + "\n\n")

        f.write("## Visualizations\n")
        f.write("The visualizations include a combined correlation heatmap and PCA scatter plot with clusters, saved in `analysis_visualization.png`.\n\n")

        f.write("## Conclusions\n")
        f.write(conclusion + "\n")

# Main function
def main(filepath):
    """
    Orchestrates the data analysis pipeline.
    """
    data = load_and_clean_data(filepath)
    eda_insights, _ = perform_eda(data)
    pca_variance = perform_clustering_and_visualization(data)

    prompt = f"Analyze the dataset columns and types:\n{data.dtypes}\nProvide insights on trends and anomalies."
    ai_insights = query_ai(prompt)

    generate_report(data, pca_variance, eda_insights, ai_insights)

if __name__ == "__main__":
    import sys
    filepath = os.getenv("DATASET_PATH", None) if len(sys.argv) < 2 else sys.argv[1]

    if not filepath:
        print("Error: Please provide a dataset path using the environment variable DATASET_PATH or as a command-line argument.")
    else:
        main(filepath)
