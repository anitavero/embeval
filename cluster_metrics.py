import argh
import re
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from prettytable import PrettyTable
from sklearn import metrics


def dbscan_clustering(model, eps=0.5, min_samples=90):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(model)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    return labels


def kmeans(model, n_clusters=3, random_state=1):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(model)
    labels = kmeans_model.labels_
    return labels


def cluster_eval(vectors, labels):
    """Unsupervised clustering metrics."""
    t = PrettyTable(['Metric', 'Score'])
    def safe_metric(metric):
        name = re.sub('_', ' ', metric.__name__).title()
        try:
            t.add_row([name, round(metric(vectors, labels), 4)])
        except ValueError as e:
            print("[{0}] {1}".format(name, e))

    safe_metric(metrics.silhouette_score)
    safe_metric(metrics.calinski_harabaz_score)
    safe_metric(metrics.davies_bouldin_score)

    print(t)


cluster_methods = {'dbscan': dbscan_clustering, 'kmeans': kmeans}


def run_clustering(model_file, cluster_method, n_clusters=3, random_state=1, eps=0.5, min_samples=90):
    if model_file == 'random':
        model = np.random.random(size=(70000, 300))
    else:
        model = np.load(model_file)

    if cluster_method == 'dbscan':
        labels = dbscan_clustering(model, eps=eps, min_samples=min_samples)
    elif cluster_method == 'kmeans':
        labels = kmeans(model, n_clusters=n_clusters, random_state=random_state)
    cluster_eval(model, labels)


if __name__ == '__main__':
    argh.dispatch_command(run_clustering)
