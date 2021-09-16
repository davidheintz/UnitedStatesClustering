import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


# method generates a plot of the clustered data during the K-means clustering method
# params: cent is the model's current centroids, pts is the dataframe containing points, col is the cluster group in pts
def gen_plot(cent, pts, col):
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'black']

    # iterate through centroids: for each, generate scatter-plot using data with matching cluster group and unique color
    for i in range(cent.shape[0]):
        plt.scatter(x=pts[pts[col] == i]['lng'],
                    y=pts[pts[col] == i]['lat'], s=1, c=colors[i])
    plt.show()


# method generates a plot of the clustered data points during the fuzzy C clustering method
# params: cent is the model's current centroids, pts dataframe contains the points
# cols contains 2 columns indices: the cluster for each point, and the % correlation to the cluster
def gen_fuzzy_plot(cent, pts, cols):
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'black']

    # storing each 4 number color variable in dataframe for correct color and transparency when plotting
    color = pts[cols[0]].apply(lambda x: colors[int(x)])  # first color becomes series of color strings
    pts['color'] = color.apply(lambda x: to_rgb(x))  # then color becomes series of RGB values
    color = pts[['color', 'max']].apply(lambda x: x[0] + (x[1],), axis=1)  # color becomes series of 3 RGBs + alpha

    plt.scatter(x=pts['lng'], y=pts['lat'], s=1, c=color)  # data is then plotted with correlated color and transparency
    plt.show()


# method searches for instance of c in points['city'] and returns the index of the found instance
# params: c is item being searched for, points is the dataframe being searched
def find_city(c, points):
    return points[points["city"] == c].index[0]


# method generates/plots k-means clusters for data continuously until no changes made from previous cluster choices
# params: seeds is a dataframe of seeds to initialize centroids, points is dataframe with data to cluster
def gen_k_means(seeds, points):

    points['cluster'] = ''
    centroids = seeds.copy()
    action = True

    while action:  # repeat while changes continue to be made
        action = False

        # iterate through points, generating euclidean distance from point to each centroid (using lat/lng)
        # cluster for data points becomes one with minimum euclidean distance (store index of cluster in point)
        for index, item in points.iterrows():
            small_dist = 1000
            cat = ''
            for idx, row in centroids.iterrows():
                # calculating euclidean distance for each point (item) with centroid (row)
                euc = np.sqrt((item['lat'] - row['lat']) ** 2 + (item['lng'] - row['lng']) ** 2)
                if euc < small_dist:
                    small_dist = euc
                    cat = idx

            # checks if new cluster is same as old: if not, store new and set action to true
            # if all clusters same as old, action remains false and while loop ends
            if points.at[index, 'cluster'] != cat:
                points.at[index, 'cluster'] = cat
                if action is False:
                    action = True

        # regenerate centroids based on cluster data (using centroids and points as param
        # then plot clusters using gen_plot where colors determined by column points['cluster']
        centroids = gen_centroids(centroids, points)
        gen_plot(centroids, points, 'cluster')
    return points


# method regenerates centroids based on params: current centroids and points dataframe
def gen_centroids(cent, pts):

    # iterate through centroid dataframe and calculate new lat/lng values
    # new lat/lng values become mean lat/lng values for all items classified in cluster
    for index, row in cent.iterrows():
        cent.at[index, 'lat'] = pts[pts['cluster'] == index]['lat'].mean()
        cent.at[index, 'lng'] = pts[pts['cluster'] == index]['lng'].mean()
    return cent


# method generates/plots fuzzy c mean clusters for data continuously until no classifications changed
# params: seeds dataframe used to initialize centroids, points dataframe used as data to train model
def gen_fuzzy_c(seeds, points):

    centroids = seeds.copy()

    # new columns in points initialized to basic integers for use later
    points['sum'] = 0  # sum of 1/(euclidean^2) for all centroids
    points['group'] = -1  # cluster group choice (index)
    points['curr'] = -1  # current centroid being checked
    points['prev'] = 0  # previous cluster group choices (to determine if classification changes occur)
    points['max'] = -1  # percent correlation for closest cluster centroid

    # continue to generate clusters and re-center the centroids until no classification changes occur
    while False in (points['prev'] == points['group']):  # if one instance of False, a change occurred
        print(centroids)
        points['prev'] = points['group'].copy()

        # iterate through centroids and calculate euclidean distances for all points in dataframe w each centroid
        # after iteration, each row in points contains a column with each 1/(euc^2) value and one with the sum of them
        # this will be used to determine the percent correlations for the points with each centroid
        for index, row in centroids.iterrows():
            euc = np.sqrt((points['lat'] - row['lat']) ** 2 + (points['lng'] - row['lng']) ** 2)  # calculating euclidean
            euc = 1 / (euc ** 2)  # using 1 / (euclidean^2) to determine correlation
            euc.replace(np.inf, 100, inplace=True)  # replace values of 1/0
            points['sum'] = points['sum'] + euc  # increment total euclidean
            points[index] = euc  # store 1/(euc^2) value at column index (related to cluster[index])

        # iterate through centroids and generate two values in points: group and max
        # group stores the cluster each point is most related to, max stores the percentage relation to that cluster
        for index, row in centroids.iterrows():

            points['curr'] = index
            points[index] = points[index] / points['sum']  # points[index] becomes percentage relation to cluster index

            # np.where used to store [current cluster index, percentage relation] in points['group','max]
            # these are only stored in instances when the current cluster relation > than the one previously stored
            cond = pd.DataFrame([points[index] > points['max'], points[index] > points['max']])
            points.loc[:, ['group', 'max']] = np.where(cond.T, points.loc[:, ['curr', index]],
                                                       points.loc[:, ['group', 'max']])

        # regenerate centroid values and then plot clusters
        centroids = gen_fuzzy_centroids(centroids, points)
        gen_fuzzy_plot(centroids, points, ['group', 'max'])
        print(points)

    return points


# method regenerates the centroid dataframe for fuzzy c clusters
# param: cent to regenerate, pts of data to use in generation
def gen_fuzzy_centroids(cent, pts):

    for index, row in cent.iterrows():
        # generate new cluster centroid by summing the pts values weighted by their relation percentage to the cluster
        per_tot = pts[index].sum()
        cent.at[index, 'lat'] = ((pts[index] / per_tot) * pts['lat']).sum()
        cent.at[index, 'lng'] = ((pts[index] / per_tot) * pts['lng']).sum()
    return cent

