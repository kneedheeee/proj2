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

    corr_matrix = data.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.savefig("analysis_visualization.png", dpi=300)
    plt.close()

# Apply PCA
def apply_pca(data, n_components=2):
    """
    Performs Principal Component Analysis (PCA) and returns the reduced data and explained variance.
    """
    numeric_data = data.select_dtypes(include=[np.number])
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(numeric_data)
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA Explained Variance: {explained_variance}")
    return reduced_data, explained_variance

# Perform clustering
def perform_clustering(data, n_clusters=3):
    """
    Performs KMeans clustering and adds cluster labels to the dataset.
    """
    numeric_data = data.select_dtypes(include=[np.number])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(numeric_data)
    print(f"Cluster Centers:\n{kmeans.cluster_centers_}")

# Query AI for insights
def query_ai(prompt, token_limit=300):
    """
    Queries an AI model for insights and returns the response.
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN', 'your_token_here')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": token_limit,
        "detail": "low"
    }

    try:
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error querying AI model: {e}"

# Generate report
def generate_report(data, pca_variance, ai_insights):
    """
    Generates a Markdown report summarizing the analysis.
    """
    with open("README.md", "w") as f:
        f.write("# Data Analysis Report\n\n")
        f.write("## Descriptive Statistics\n")
        f.write(data.describe().to_markdown())
        f.write("\n\n")

        f.write("## PCA Explained Variance\n")
        f.write(f"{pca_variance}\n\n")

        f.write("## AI Insights\n")
        f.write(ai_insights)
        f.write("\n\n")

        f.write("## Visualizations\n")
        f.write("![Analysis Visualization](analysis_visualization.png)\n")

# Main function
def main(filepath):
    """
    Orchestrates the data analysis pipeline.
    """
    data = load_and_clean_data(filepath)
    perform_eda(data)
    reduced_data, pca_variance = apply_pca(data)
    perform_clustering(data)

    prompt = f"Analyze the dataset columns and types:\n{data.dtypes}\nProvide insights on trends and anomalies."
    ai_insights = query_ai(prompt)

    generate_report(data, pca_variance, ai_insights)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset_path>")
    else:
        main(sys.argv[1])
