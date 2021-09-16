# UnitedStatesClustering
This program uses a dataset downloaded from simplemaps.
(url: https://simplemaps.com/data/us-cities)
This dataset contains information about 28,372 US cities.
The information used will be the latitude and longitude. 

This program was designed by myself, David Heintz in an attempt to create clustering models and observe the application of these models attempting to fit a dataset of longitude and latitude coordinates for 10000 United States cities

The methods I have designed so far are the k-means method and the fuzzy-c-means method. Instead of a random seed, both my models use a seed of the 8 cities in the united states with the highest populations: "New York", "Los Angeles", "Houston", "Chicago", "Philadelphia", "Phoenix", "San Antonio", "San Diego"

Using this seed, both methods fit a cluster model for the cities data based on the euclidean distance to each centroid (initialized as seed). The k-means method is hard fitted meaning each data point is only in one cluster, while the fuzzy-c-means method is soft fitted meaning each data point has a percent relation to each cluster. The clusters are plotted and refit repeatedly until there are no more possible cluster changes 

Observations:
- k-means: iteration through dataframe of >= 10000 rows is inefficient
- ^ solution: iterate through centroids only and use dataframe functions
- k-means: self made function with predetermined seeds fitted a similar model to the sklearn model. each cluster had similar size and weight
- fuzzy-c-means: iterating through centroids only improved efficiency over k-means function, even though more calculations are performed
- fuzzy-c-means: since centroids regenerated based on positions of all items and their % relation, all centroids drift towards the center after many refittings. (drifting towards center causes getting closer to opposite side, getting closer to opposite side causes drifting towards center) Each cluster retains most original members, but with a lower relation percentage.
- comparison: k-means clusters had similar shape and size, while fuzzy-c-mean had all different shapes with groups spread across the map and eventually many clusters forming within other clusters. 
- comparison: k-means also reached an endpoint with no further changes possible after less than 30 maps generated, fuzzy-c-means however continued to produce a new map every couple seconds for over and hour and was terminated before an end was reached.
