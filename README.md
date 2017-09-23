# KMeans
KMeans machine learning algorithm displaying Voronoi graph for Fisher's Iris data set

When executing the algorithm, the console will display the centroids and the new ones changed. If wanted, the number of centroids and the threshold can be changed in the code.

Once the centroid locations are permenant, then a Voronoi graph is displayed for the first run Kmeans algorithm.

After that, the algorithm is run multiple times, using different split points for the training and testing datasets. For each split point value, the algorithm is run test_run times. For each run, a confusion matrix is printed in the console alongside an accuracy report which includes precision, recall, f1-score and support. 
