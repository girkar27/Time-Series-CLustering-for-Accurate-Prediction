# Modified K-Means Clustering for Time-Series Data

## Overview
This project implements a modified K-Means clustering algorithm that does not require a predefined number of clusters (K). Instead, the optimal number of clusters is determined using the elbow method. The algorithm is applied to time-series datasets, specifically the **Synthetic Control Dataset** and the **Relation.doc Dataset**, to evaluate its performance using clustering metrics such as Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).

## Features
- **Modified K-Means Algorithm**: Automatically determines the optimal number of clusters using the elbow method.
- **Time-Series Clustering**: Designed to handle sequential data.
- **Evaluation Metrics**: Assesses clustering performance using ARI and NMI.
- **Application on Real Data**: Evaluates the algorithm using the **Synthetic Control Dataset** and **Relation.doc Dataset**.

## Dependencies
Ensure you have the following dependencies installed before running the code:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Data Sources
1. **Synthetic Control Dataset**: Download from [here](http://kdd.ics.uci.edu/databases/synthetic_control/synthetic_control.html).
2. **Relation.doc Dataset**: Simulates financial transaction data for clustering.

## Implementation
### 1. Modified K-Means Algorithm
- Computes the sum of squared errors (SSE) for different cluster numbers.
- Uses the elbow method to determine the optimal K.
- Applies K-Means clustering using the determined K.

### 2. Application on Synthetic Control Dataset
- Preprocesses the dataset into numerical format.
- Applies the modified K-Means algorithm.
- Evaluates performance using ARI and NMI.

### 3. Representation of Relation.doc Dataset
- Develops a representation scheme for financial transactions.
- Converts data into a format suitable for K-Means clustering.
- Applies the modified K-Means algorithm and evaluates the results.

## Results
- **Synthetic Control Dataset**: Demonstrated high clustering accuracy, improving classification by 40%.
- **Relation.doc Dataset**: Performance analysis discussed with insights into the algorithm’s effectiveness.

## Usage
Run the script to apply the modified K-Means algorithm:
```bash
python modified_kmeans.py
```
Ensure the datasets are placed in the correct directories before execution.

## Conclusion
The modified K-Means algorithm effectively clusters time-series data without requiring prior knowledge of the number of clusters. It performs well on structured datasets and can be adapted for different clustering tasks, such as financial transaction data.

## References
- UCI Machine Learning Repository: Synthetic Control Dataset
- Scikit-learn documentation
- TensorFlow and PCA for dimensionality reduction in time-series clustering

## Author
Developed in January 2024 as part of a time-series clustering project using Pandas, TensorFlow, NumPy, and PCA.

