from sklearn.cluster import KMeans


def performClustering(sentences):
    kmeans = KMeans(n_clusters=10, random_state=0).fit(sentences)
    print(kmeans)
    return kmeans