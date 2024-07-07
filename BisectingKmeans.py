# Libraraies used to perform Kmeans function
import os
import numpy as np
import matplotlib.pyplot as plt
from Kmeans import kmeans

# Ensures the random number generation will produce the same sequence of random numbers 
np.random.seed(42)

# This function is used to compute the distance between two points present
def compute_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def initial_selection_random(data, k):
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

def compute_sum_of_square(cluster_points, centroid):
    """
    This function basically finds the sum of the squares of the distances from each point in a group to its center. 
    Imagine you have a bunch of points scattered around, and  measure how spread out they are from the middle of the group. 
    measuring how far it is from the middle (the centroid), squaring that distance, 
    and then adding up all these squared distances to give you a single number that represents the overall spread of the points.
    """
    return np.sum([compute_distance(point, centroid) ** 2 for point in cluster_points])

def bisecting_kmeans(data, k):
    """
    This function takes a bunch of data points and splits them into groups, trying to make 'k' clusters out of them. 
    Then, it returns which cluster each data point belongs to.
    """
    cluster_ids = np.zeros(len(data))
    centroids = initial_selection_random(data, 1)
    # Iteratively split clusters until 'k' clusters are obtained
    for i in range(1, k):
        cluster_ids_temp = assign_cluster_ids(data, centroids)
        cluster_id_to_split = np.argmax([np.sum(cluster_ids_temp == j) for j in range(i)])
        cluster_points = data[cluster_ids_temp == cluster_id_to_split]
        # Perform k-means clustering to split the selected cluster
        centroids_split, _ = kmeans(cluster_points, 2)
        cluster_ids[cluster_ids_temp == cluster_id_to_split] = i * np.ones(len(cluster_points))
        # Update centroids
        centroids = np.concatenate((centroids[:cluster_id_to_split], centroids_split, centroids[cluster_id_to_split+1:]), axis=0)
    return cluster_ids


def compute_silhouette_coefficient(data, cluster_ids, k):
    """
    This function computes the average silhouette coefficient for a clustering, assessing how well each data point fits its assigned cluster. 
    It takes in the data points, their cluster IDs, and the number of clusters. 
    The output is a single number representing the average silhouette coefficient across all data points.
    """
    silhouette_coefficients = []
    for i, point in enumerate(data):
        cluster_id = cluster_ids[i]
        cluster_points = data[cluster_ids == cluster_id]
        # Skip computation if the cluster is empty
        if len(cluster_points) == 0:
            continue  
        
        # Exclude empty clusters
        other_cluster_points = [data[cluster_ids == j] for j in range(k) if j != cluster_id]
        other_cluster_points = [ocp for ocp in other_cluster_points if len(ocp) > 0]  
        
        # Skip computation if there are no other clusters available
        if len(other_cluster_points) == 0:
            continue  
        
        intra_cluster_distance = np.mean([compute_distance(point, p) for p in cluster_points])
        nearest_cluster_distance = np.min([np.mean([compute_distance(point, p) for p in other_cluster]) for other_cluster in other_cluster_points])
        
        # Skip computation if intra-cluster distance or nearest cluster distance is NaN
        if np.isnan(intra_cluster_distance) or np.isnan(nearest_cluster_distance):
            continue  
        
        silhouette_coefficient = (nearest_cluster_distance - intra_cluster_distance) / max(intra_cluster_distance, nearest_cluster_distance)
        silhouette_coefficients.append(silhouette_coefficient)
    
    return np.mean(silhouette_coefficients)

# Function to plot number of clusters vs. silhouette coefficient values
def plot_silhouette(k_values, silhouette_scores):
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette coefficient')
    plt.title('Silhouette Coefficient for Different Values of k')
    plt.xticks(range(1, len(k_values) + 1))  
    plt.grid()
    plt.show()

def load_dataset(filename):
    try:
        # Get the current directory
        current_dir = os.getcwd()
        # Joining the current directory with the filename
        filepath = os.path.join(current_dir, filename)
        dataset = np.loadtxt(filepath, delimiter=' ', dtype=str)
        # Extracting only the numerical features from the dataset
        dataset_numeric = dataset[:, 1:].astype(float)
        return dataset_numeric
    except Exception as e:
        print("Error loading dataset:", e)
        return None

def main():
    dataset = load_dataset('dataset')    
    if dataset is not None:
        print("Dataset loaded successfully.")
    else:
        print("Failed to load dataset. Please check the filename and file path.")
        return

    k_values = range(1, 11)  

    silhouette_scores = []

    for k in k_values:
        cluster_ids = bisecting_kmeans(dataset, k)
        silhouette_avg = compute_silhouette_coefficient(dataset, cluster_ids, k)
        silhouette_scores.append(silhouette_avg)
        # Print silhouette coefficient for each k
        print(f"Silhouette coefficient for k = {k}: {silhouette_avg}")  

    plot_silhouette(k_values, silhouette_scores)

if __name__ == "__main__":
    main()
