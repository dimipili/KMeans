# K Means clustering algorithm for any data set, uses the iris flower data set as an example
import math, random
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def read_file(fname):
    lines = []
    with open(fname, 'r') as f:  # opens the file and then closes it once it is done with it
        for l in f:
            ls = l.strip().split(',') #removes any whitespaces and splits the values with commas
            ls = list(map(float,ls[:-1])) + [ls[-1]] #makes all of the lines in the list float numbers
            lines.append(ls) #appends the line into the list
    return lines

def init_centroids(points, n_clusters): #picks N random centroids from a list of points
    return random.sample(points,n_clusters)

def eu_dist(v1, v2): #v1,v2 are vectors
    l1 = [((s[0] - s[1]) ** 2) for s in zip(v1,v2)]  # it finds the difference in the coordinates, and then zip basically combines the two vectors together into a list
    return math.sqrt(reduce(lambda v1, v2: v1 + v2, l1,0))

def distances(vec, centroids): #finds distances to the centroids
    return [eu_dist(c, vec) for c in centroids]

def min_pos(distances): #finds the minimum position to a certain centroid
    return reduce(lambda x, y: x if x[0] < y[0] else y, zip(distances, range(len(distances))))[1] #finds the minimum position to a centroid for each point and returns the cluster it belongs to

def find_clust(centroids, vec): #determines the shortest distance to a cluster for one vector
    dists = distances(vec, centroids)
    minpos = min_pos(dists)
    return minpos

def cluster_all(centroids, vecs): #puts all the points in their corresponding clusters
    clusters = []
    for i in range(len(centroids)):
        clusters.append(list())
    for vec in vecs:
        mp = find_clust(centroids, vec)
        clusters[mp].append(vec)
    return clusters

def add_vecs(v1,v2): #adds 2 vectors
    return [v[0]+v[1] for v in zip(v1,v2)]

def mean_of_vecs(vecs): #finds the mean of a list of vectors
    sum_of_vecs = reduce(add_vecs,vecs)
    return [q/len(vecs) for q in sum_of_vecs]

def new_centroids(clusters): #calculates the new centroids by finding the mean position from all the points in a cluster
    newCentroids = []
    for c in clusters:
        mean = mean_of_vecs(c)
        newCentroids.append(mean)
    return newCentroids

def change_in_centroid(old_centroids, new_centroids, threshold): #checks if there was a change in position in any of the centroid of clsuters by a certain threshold
    A = [eu_dist(v[0],v[1]) for v in zip(old_centroids, new_centroids)]
    print(A)
    survived = list(filter(lambda x: x > threshold,A)) #checks if all the changes in centroids is larger or smaller than a threshold
    if len(survived) == 0: #empty list means that there wasn't a change
        change = False
        return change
    else:
        change = True
        return change

def run_iris_dataset(vectors): #calls all the methods into one larger function
    change = True #will continue to run until there is no change in centroids or a change smaller than the threshold
    initial_centroids = init_centroids(vectors, 3)  # determines the initial centroids
    while change == True:
        clusters = cluster_all(initial_centroids, vectors)   # puts all the vectors into their clusters
        new_cents = new_centroids(clusters)  # determines the new clusters
        change = change_in_centroid(initial_centroids, new_cents, 0.001)  # threshold is 0.01
        if change == False:
            print("The algorithm is finished and the graph will be displayed now.")
            break
        initial_centroids = new_cents
    return initial_centroids

def obtain_data():
    avectors = read_file('iris.csv')
    avectors1 = [x[:-1] for x in avectors]

    centers1 = run_iris_dataset(avectors1)
    clusters1 = cluster_all(centers1, avectors1) #clusters from the program
    obtain_data_tup = (avectors,avectors1,centers1,clusters1)
    return obtain_data_tup

def single_confusion_matrix_and_accuracy(avectors,centers1):
    fake_name_map2 = {'Iris-virginica':0, 'Iris-versicolor':1, 'Iris-setosa':2}

    actual = [fake_name_map2[q[-1]] for q in avectors]
    pred = [find_clust(centers1, q[:-1]) for q in avectors]
    print("predicted:",pred)
    acc_score = accuracy_score(actual, pred)
    print("The accuracy is {acc}%".format(acc=acc_score*100))
    class_names = ["Iris-setosa","Iris-versicolor","Iris-virginica"]

    conf_matrix = print(confusion_matrix(actual, pred))
    print(classification_report(actual,pred,target_names=class_names))
    confusion_matrix_tup = (acc_score,conf_matrix)
    return confusion_matrix_tup

def items(avectors1):
    train = avectors1
    iris_centroids = run_iris_dataset(train) #centroids from the program
    iris_clusters = cluster_all(iris_centroids, train) #clusters from the program
    items_tup = (iris_centroids,iris_clusters,train)
    return items_tup

def graph_dimension_reduction(train):
    reduced_data = PCA(n_components=2).fit_transform(train) #bascially plots the 4d points onto a 2d graph
    reduced_iris_centroids = run_iris_dataset(reduced_data.tolist())
    graph_dimension_reduction_tup = (reduced_data,reduced_iris_centroids)
    return graph_dimension_reduction_tup

def predict_all(centers, dat):
    ret = np.ndarray((len(dat), ))
    for i in range(len(dat)):
        ret[i] = find_clust(centers, dat[i])
    return ret

def graph_dimensions(reduced_iris_centroids):
    h = 0.01 #decrease the value to increase the quality of the graph
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = predict_all(reduced_iris_centroids, np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    print('xx shape: ',xx.shape)
    plt.figure(1)
    plt.clf()
    graph_dimensions_tup = (Z,xx,yy,reduced_iris_centroids,x_min,x_max,y_min,y_max)
    return graph_dimensions_tup

def display_map(Z,xx,yy,reduced_data,reduced_iris_centroids,x_min,x_max,y_min,y_max):
    plt.imshow(Z, interpolation='none',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

    # Plot the centroids as a white X
    centroids = np.asarray(reduced_iris_centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering algorithm using the Iris dataset\n'
              'Centroids are marked with white cross, data points are marked with black dots')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    #plt.savefig('K-means_algorithm_Voronoi_graph.png')
    plt.show()

#first example is so that all of it works and to show what one graph looks like
obtained_data = obtain_data()
avectors = obtained_data[0]
avectors1 = obtained_data[1]
centers1 = obtained_data[2]
clusters1 = obtained_data[3]

individual_confusion_matrix = single_confusion_matrix_and_accuracy(avectors,centers1) #,avectors1)
acc_score = individual_confusion_matrix[0]
conf_matrix = individual_confusion_matrix[1]

items = items(avectors1)
iris_centroids = items[0]
iris_clusters = items[1]
train = items[2]

graph_dimensions_reduction = graph_dimension_reduction(train)
reduced_data = graph_dimensions_reduction[0]
reduced_iris_centroids = graph_dimensions_reduction[1]

single_graph_dimensions = graph_dimensions(reduced_iris_centroids)
Z = single_graph_dimensions[0]
xx = single_graph_dimensions[1]
yy = single_graph_dimensions[2]
reduced_iris_centroids = single_graph_dimensions[3]
x_min = single_graph_dimensions[4]
x_max = single_graph_dimensions[5]
y_min = single_graph_dimensions[6]
y_max = single_graph_dimensions[7]

display_map(Z,xx,yy,reduced_data,reduced_iris_centroids,x_min,x_max,y_min,y_max)

def main_fuction(avectors):
    split_point = [30,75,90,105,120,135]  # 20%,50%,60%,70%,80%,90% training set 30% validation set  #random.randint(40, 89)  # returns a value between 40 and 90
    for x in range(len(split_point)): #as to vary the split points and get different results
        random.shuffle(avectors)
        test_runs = 10
        training_set = [x[:-1] for x in avectors[0:split_point[x]]]
        validation_set = avectors[split_point[x]:]
        print("1",validation_set)
        for run in range(test_runs):
            random.shuffle(training_set)
            random.shuffle(validation_set)
            centers1 = run_iris_dataset(training_set)
            single_confusion_matrix_and_accuracy(validation_set,centers1)

main_fuction(read_file('iris.csv'))