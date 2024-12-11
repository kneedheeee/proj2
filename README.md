# Data Analysis Report

## Overview

This Python-based data analysis script automates several key steps in data exploration and visualization. The script loads, cleans, analyzes, and visualizes a dataset using various techniques including PCA (Principal Component Analysis), KMeans clustering, correlation heatmaps, and AI-driven insights. 

The analysis generates reports and visualizations which are then saved in various formats, including PNG images for correlation heatmaps and cluster plots. Additionally, advanced insights are derived through querying an AI model.

## Features

1. **Data Loading with Encoding Fallback**:  
   The dataset is loaded with an encoding fallback system to prevent errors when dealing with different encodings, ensuring a smooth data loading process.

2. **Correlation Heatmap**:  
   A heatmap is generated to visualize the correlations between numerical features in the dataset, making it easier to identify relationships between different variables.

3. **PCA (Principal Component Analysis)**:  
   PCA is applied to reduce the dimensionality of the dataset, providing a simplified view of the dataset's key components. The script outputs the explained variance for each principal component.

4. **KMeans Clustering**:  
   KMeans clustering is performed to group the data into a specified number of clusters (default is 3). A scatter plot is generated to visualize the clusters.

5. **AI-driven Insights**:  
   The script integrates with an AI model to generate insights about the dataset, identifying trends, anomalies, and other interesting patterns.

6. **Data Cleaning**:  
   Rows with missing values are removed, and outliers are detected using z-scores. The cleaned dataset is used for further analysis.

7. **Markdown Report Generation**:  
   A comprehensive report is automatically generated in Markdown format, summarizing the dataset's structure, insights, and visualizations.

## Requirements

- Python 3.9 or higher
- Libraries:
  - pandas
  - seaborn
  - matplotlib
  - numpy
  - scipy
  - openai
  - scikit-learn
  - requests

You can install the required libraries using:

```bash
pip install pandas seaborn matplotlib numpy scipy openai scikit-learn requests
```

## Usage

1. Clone or download this script.
2. Ensure the necessary libraries are installed.
3. Run the script by providing the dataset path as an argument:

```bash
python autolysis.py <dataset_path>
```

The script will process the dataset, generate visualizations, AI insights, and save the results as files (`correlation_heatmap.png`, `cluster_plot.png`, and `README.md`).

## Output

- **correlation_heatmap.png**: A heatmap visualization of the correlation between numerical features in the dataset.
- **cluster_plot.png**: A scatter plot visualizing the clusters created by the KMeans algorithm.
- **README.md**: A Markdown report summarizing the dataset, analysis, insights, and visualizations.

## Functions

### `load_data_with_fallback(filepath)`
- Loads the dataset from the specified file path, attempting to read with UTF-8 encoding first, and falling back to Latin-1 if UTF-8 fails.

### `create_correlation_heatmap(data, save_as)`
- Generates a heatmap of correlations for numerical features and saves it to the specified file path.

### `apply_pca(data, components=2)`
- Performs PCA on the dataset, returning the transformed data and the explained variance.

### `perform_kmeans_clustering(data, plot_filename)`
- Performs KMeans clustering on the dataset and visualizes the clusters in a scatter plot, saving it to the specified file path.

### `query_ai_model(prompt, token_limit=300)`
- Sends a prompt to an AI model via an AI Proxy service, and returns the response.

### `clean_data_and_find_outliers(dataset)`
- Cleans the dataset by removing rows with missing values and detecting outliers using z-scores.

### `run_analysis(file_path)`
- The main function orchestrating the entire analysis process, including loading the dataset, cleaning the data, applying PCA, performing clustering, generating AI insights, and creating the Markdown report.

## Example

```bash
python autolysis.py data.csv
```

This command will process the dataset `data.csv`, perform the analysis, and generate visualizations and insights, with outputs saved as `correlation_heatmap.png`, `cluster_plot.png`, and `README.md`.
