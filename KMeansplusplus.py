# Libraraies used to perform Kmeans function
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensures the random number generation will produce the same sequence of random numbers 
np.random.seed(42) 

# This function is used to compute the distance between two points present 
def compute_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def initial_selection_kmeanspp(data, k):
    """"
    This function randomly selects one datapoint from dataset as the first centeroid and then calaculates the distance to the nearest centeroid. 
    Then it will compute the distance and probability for each data point being chosen as the next centeroid. 
    """ 
    centroids = [data[np.random.randint(0, len(data))]]
    while len(centroids) < k:
        distances = np.array([min([compute_distance(c, point) for c in centroids]) for point in data])
        probabilities = distances / np.sum(distances)
        cumulative_probabilities = np.cumsum(probabilities)
        rand_val = np.random.rand()
        chosen_index = np.argmax(cumulative_probabilities >= rand_val)
        centroids.append(data[chosen_index])
    return np.array(centroids)


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
    This function will compute the silhoutte coefficient for all data points in a clustering result, later the silhoutte score will be appended to a list of coefficients.
    If the cluster is empty or all other clusters are empty, then it will skip the computation for this data point. 
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
 

def kmeans(data, k, max_iter=100, initialization='random'):
    """
    This function implements the K means algorithm. It will select the initial centeroids and iterate, later assigining each data point 
    to the nearest centeroid. 
    """
    if initialization == 'random':
        centroids = initial_selection_random(data, k)
    elif initialization == 'kmeans++':
        centroids = initial_selection_kmeanspp(data, k)
    else:
        raise ValueError("Initialization method not supported. Use 'random' or 'kmeans++'.")
    
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
    plt.title('Silhouette Coefficient for KMeansplusplus') 
    plt.xticks(range(1, len(k_values) + 1))  
    plt.grid() 
    plt.show()


def load_dataset(filename):
    """
    This function is used to load the dataset in text format and we need to make sure the file is present in the same directory.
    Then it will load the dataset using numpy and will extract only the numeric values present. 
    """
    try:
        # Get the current directory
        current_dir = os.getcwd()
        # Join the current directory with the filename
        filepath = os.path.join(current_dir, filename)
        dataset = np.loadtxt(filepath, delimiter=' ', dtype=str)
        # Extracting only the numerical features from the dataset
        dataset_numeric = dataset[:, 1:].astype(float)
        return dataset_numeric
    except Exception as e:
        print("Error loading dataset:", e)
        return None

"""
This function is used to load the dataset and will print a success message if data is loaded properly. 
Also initializes k values ranging from 1 to 10
It will iterate over each values of k using k means algorithm 
It calculates the silhouette coefficient for the clustering
Outputs the silhoutte coefficient 
""" 
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
        centroids, cluster_ids = kmeans(dataset, k, initialization='kmeans++')  
        silhouette_avg = compute_silhouette_coefficients(dataset, cluster_ids, k)
        silhouette_scores.append(silhouette_avg)
        print(f"Silhouette coefficient for k = {k}: {silhouette_avg}") 

    plot_silhouette(k_values, silhouette_scores)

    # Add other functionalities as required

if __name__ == "__main__":
    main() 
 