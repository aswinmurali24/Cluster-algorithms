Description
This project implements and compares various clustering algorithms using the provided dataset. The algorithms evaluated are k-means, k-means++, and Bisecting k-Means, with performance measured by the Silhouette coefficient for different values of 
𝑘
k. Synthetic data of the same size is also generated and clustered for comparison.

Tasks
K-Means Clustering:

Implemented k-means clustering.
Varied 
𝑘
k from 1 to 9.
Computed the Silhouette coefficient for each set of clusters.
Plotted 
𝑘
k vs. Silhouette coefficient.
Synthetic Data Clustering:

Generated synthetic data with the same number of data points as the provided dataset.
Applied k-means clustering to the synthetic data.
Plotted 
𝑘
k vs. Silhouette coefficient for synthetic data.
K-Means++ Clustering:

Implemented k-means++ clustering.
Varied 
𝑘
k from 1 to 9.
Computed the Silhouette coefficient for each set of clusters.
Plotted 
𝑘
k vs. Silhouette coefficient.
Bisecting k-Means Clustering:

Implemented Bisecting k-Means to create a hierarchy of clusters.
Extracted clustering with 
𝑠
s clusters for 
𝑠
s from 1 to 9.
Computed the Silhouette coefficient for each clustering.
Plotted 
𝑠
s vs. Silhouette coefficient.
Results
Each clustering algorithm's performance is visualized by plotting the Silhouette coefficient against the number of clusters. These plots provide insights into the clustering quality and help determine the optimal number of clusters.


