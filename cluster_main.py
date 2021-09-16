from sklearn.cluster import KMeans
import pandas as pd
import cluster_engine as ce

# store cities from excel file into dataframe, eliminate unnecessary columns and only keep first 10000 rows
# using lat and lng values for calculating euclidean distance and clustering
cities = pd.read_excel("uscities.xlsx")
cols = ["city", "lat", "lng", "state_name"]
cities = cities[cols]
cities = cities.iloc[:10000, ]

# eliminate states not land connected (messes with centroid positioning)
indices = cities[cities['state_name'].isin(['Alaska', 'Hawaii', 'Puerto Rico'])].index
cities.drop(indices, axis=0, inplace=True)

# determine seed cities (largest populations) and store in seeds dataframe
big_cities = ["New York", "Los Angeles", "Houston", "Chicago", "Philadelphia", "Phoenix", "San Antonio", "San Diego"]
positions = []
for city in big_cities:
    positions.append(ce.find_city(city, cities))
seeds = cities.loc[positions]

# convert dataset to numpy and generate/fit k-means model clustering using auto sklearn method (random seeds)
coord = cities[['lat', 'lng']].to_numpy()
k_means = KMeans(n_clusters=8, init='random').fit(coord)
cities['k_mean'] = k_means.labels_

seeds.reset_index(drop=True, inplace=True)
ce.gen_plot(seeds, cities, 'k_mean')

# generate/plot k-means and fuzz-c means clusters for cities using non random seeds and self made methods
cities = ce.gen_k_means(seeds, cities)
cities = ce.gen_fuzzy_c(seeds, cities)

print(cities)

# TODO: compare cities['cluster'] with cities['group'] and see how many match
