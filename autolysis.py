# /// script
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "scikit-learn",
#   "numpy",
#   "scipy",
#   "requests<3",
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


def load_data_with_fallback(filepath):
    """
    Attempts to load a dataset with UTF-8 encoding, falling back to Latin-1 if UTF-8 fails.
    """
    try:
        return pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding="latin-1")


def create_combined_visualization(data, save_as):
    """
    Generates a combined visualization with a correlation heatmap and KMeans clustering scatter plot.
    The output is saved as a single PNG file.
    """
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Correlation heatmap
    corr_matrix = data.select_dtypes(exclude="object").corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Correlation Coefficient"},
        ax=axes[0]
    )
    axes[0].set_title("Correlation Heatmap")

    # KMeans clustering
    numeric_data = data.select_dtypes(include=[np.number]).dropna()
    if not numeric_data.empty:
        kmeans_model = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans_model.fit_predict(numeric_data)
        axes[1].scatter(
            numeric_data.iloc[:, 0],
            numeric_data.iloc[:, 1],
            c=cluster_labels, cmap="viridis", alpha=0.7
        )
        axes[1].set_title("KMeans Clustering (k=3)")
        axes[1].set_xlabel(numeric_data.columns[0])
        axes[1].set_ylabel(numeric_data.columns[1])
    else:
        axes[1].text(0.5, 0.5, "No numeric data available for clustering", horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)

    # Save the combined visualization
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.close()


def apply_pca(data, components=2):
    """
    Performs PCA on numeric columns of the dataset, reducing dimensions and returning results and variance explained.
    """
    numeric_data = data.select_dtypes(include=[np.number]).dropna()
    pca_model = PCA(n_components=components)
    reduced_data = pca_model.fit_transform(numeric_data)
    variance_explained = pca_model.explained_variance_ratio_
    return reduced_data, variance_explained


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
        "max_tokens": token_limit,
        "detail": "low"  # Reduce cost by specifying low detail
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


def clean_data_and_find_outliers(dataset):
    """
    Cleans data by removing rows with NaN values and detects outliers using z-scores.
    """
    cleaned_data = dataset.dropna()
    z_scores = np.abs(zscore(cleaned_data.select_dtypes(include=[np.number])))
    outlier_mask = (z_scores > 3).any(axis=1)
    cleaned_data = cleaned_data[~outlier_mask]
    return cleaned_data, outlier_mask


def run_analysis(file_path):
    """
    The main function orchestrating data analysis and reporting.
    """
    dataset = load_data_with_fallback(file_path)

    # Clean and analyze data
    cleaned_data, outliers = clean_data_and_find_outliers(dataset)
    print(f"Outliers detected: {np.sum(outliers)}")

    # PCA analysis
    pca_results, variance = apply_pca(cleaned_data)

    # Generate combined visualization
    combined_plot_path = "analysis_visualization.png"
    create_combined_visualization(cleaned_data, combined_plot_path)

    # AI insights
    ai_prompt = f"Analyze dataset columns and trends:\n{cleaned_data.dtypes}"
    ai_insights = query_ai_model(ai_prompt)

    # Generate Markdown report
    with open("README.md", "w") as f:
        f.write("# Data Analysis Report\n\n")
        f.write(f"Dataset contains {len(dataset)} rows and {len(dataset.columns)} columns.\n")
        f.write(f"Outliers detected: {np.sum(outliers)}\n")
        f.write("## PCA Explained Variance\n")
        f.write(f"{variance}\n")
        f.write("## Insights\n")
        f.write(ai_insights)
        f.write("\n\n## Visualization\n")
        f.write(f"![Combined Visualization]({combined_plot_path})\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset_path>")
        sys.exit(1)
    run_analysis(sys.argv[1])
