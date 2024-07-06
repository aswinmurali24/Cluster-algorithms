Description
This project implements and compares various clustering algorithms using the provided dataset. The algorithms evaluated are k-means, k-means++, and Bisecting k-Means, with performance measured by the Silhouette coefficient for different values of k. Synthetic data of the same size is also generated and clustered for comparison.

Tasks

K-Means Clustering:

Implemented k-means clustering.
Varied 𝑘 from 1 to 9.
Computed the Silhouette coefficient for each set of clusters.
Plotted k vs. Silhouette coefficient.


Synthetic Data Clustering:

Generated synthetic data with the same number of data points as the provided dataset.
Applied k-means clustering to the synthetic data.
Plotted k vs. Silhouette coefficient for synthetic data.


K-Means++ Clustering:

Implemented k-means++ clustering.
Varied k from 1 to 9.
Computed the Silhouette coefficient for each set of clusters.
Plotted k vs. Silhouette coefficient.


Bisecting k-Means Clustering:

Implemented Bisecting k-Means to create a hierarchy of clusters.
Extracted clustering with s clusters for s from 1 to 9.
Computed the Silhouette coefficient for each clustering.
Plotted s vs. Silhouette coefficient.


Results
Each clustering algorithm's performance is visualized by plotting the Silhouette coefficient against the number of clusters. These plots provide insights into the clustering quality and help determine the optimal number of clusters.


