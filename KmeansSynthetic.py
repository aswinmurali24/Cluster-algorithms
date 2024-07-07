# Libraraies used to perform Kmeans function 
import numpy as np
import matplotlib.pyplot as plt
# Importing the kmeans function from KMeans.py
from Kmeans import kmeans   

# Ensures the random number generation will produce the same sequence of random numbers 
np.random.seed(42)


def generate_synthetic_data(size):
    """
    This function is used generate synthetic data. It also creates an array of random numbers sampled from a uniform distribution over the interval [0, 1].
    The data has a shape of (size, 2). 
    """ 
    synthetic_data = np.random.rand(size, 2)  
    return synthetic_data

# This function is used to compute the distance between two points present 
def compute_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def initial_selection(data, k):
    """
    This function will randomly generate k indicies and then extracts the data points from the dataset to form initial centeroids
    """
    centroids_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[centroids_indices]
    return centroids


def assign_cluster_ids(data, centroids):
    """
    This function will assign cluster ID's to each data point based on their distances to the centeroids of the clusters 
    """ 
    cluster_ids = []
    for point in data:
        distances = [compute_distance(point, centroid) for centroid in centroids]
        closest_cluster = np.argmin(distances)
        cluster_ids.append(closest_cluster)
    return np.array(cluster_ids)


def compute_cluster_representatives(data, cluster_ids, k):
    """
    This function will compute the clusters representatives for each and every cluster in the dataset. 
    This will output an array containing the computed cluster representatives. 
    """ 
    cluster_representatives = []
    for cluster_id in range(k):
        cluster_points = data[cluster_ids == cluster_id]
        cluster_representative = np.mean(cluster_points, axis=0)
        cluster_representatives.append(cluster_representative)
    return np.array(cluster_representatives)


def compute_silhouette_coefficient(point, cluster_id, cluster_points, other_cluster_points):
    """
    This function will calculate the silhoutte coeffcient for a single data point within the clustering algorithm.
    It will also calculate the average distance from input point to other points in the same cluster and also the neighbouring clusters
    """ 
    intra_cluster_distance = np.mean([compute_distance(point, p) for p in cluster_points])
    nearest_cluster_distance = np.min([np.mean([compute_distance(point, p) for p in other_cluster]) for other_cluster in other_cluster_points])
    silhouette_coefficient = (nearest_cluster_distance - intra_cluster_distance) / max(intra_cluster_distance, nearest_cluster_distance)
    return silhouette_coefficient


def compute_silhouette_coefficients(data, cluster_ids, k):
    """
    This function will compute the clusters representatives for each and every cluster in the dataset. 
    This will output an array containing the computed cluster representatives. 
    """ 
    silhouette_coefficients = []
    for i, point in enumerate(data):
        cluster_id = cluster_ids[i]
        cluster_points = data[cluster_ids == cluster_id]
        if len(cluster_points) == 0:
            continue  
        
        other_cluster_points = [data[cluster_ids == j] for j in range(k) if j != cluster_id]
        if all(len(ocp) == 0 for ocp in other_cluster_points):
            continue  
        
        silhouette_coefficient = compute_silhouette_coefficient(point, cluster_id, cluster_points, other_cluster_points)
        silhouette_coefficients.append(silhouette_coefficient)
    return np.mean(silhouette_coefficients)


def kmeans(data, k, max_iter=100):
    """
    This function implements the K means algorithm. It will select the initial centeroids and iterate, later assigining each data point 
    to the nearest centeroid. 
    """
    centroids = initial_selection(data, k)
    for _ in range(max_iter):
        cluster_ids = assign_cluster_ids(data, centroids)
        new_centroids = compute_cluster_representatives(data, cluster_ids, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, cluster_ids

"""
This function is to plot number of clusters vs silhouette coefficient values
"""
def plot_silhouette(k_values, silhouette_scores):
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette coefficient')
    plt.title('Silhouette Coefficient for K Means(Synthetic Data)') 
    plt.xticks(range(1, len(k_values) + 1))
    plt.grid() 
    plt.show()

"""
This function is used to load the dataset and will print a success message if data is loaded properly. 
Also initializes k values ranging from 1 to 10
It will iterate over each values of k using k means algorithm 
It calculates the silhouette coefficient for the clustering
Outputs the silhoutte coefficient 
""" 
def main():
    # Generating synthetic data of the same size as the provided dataset
    dataset_size = 100  
    synthetic_data = generate_synthetic_data(dataset_size)

    k_values = range(1, 11)  
    silhouette_scores = []

    for k in k_values:
        centroids, cluster_ids = kmeans(synthetic_data, k)  
        # Compute silhouette coefficient and append to list
        silhouette_avg = compute_silhouette_coefficients(synthetic_data, cluster_ids, k)
        silhouette_scores.append(silhouette_avg)
        print(f"Silhouette coefficient for k = {k}: {silhouette_avg}")  

    plot_silhouette(k_values, silhouette_scores)

if __name__ == "__main__":
    main() 
