import warnings

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.cluster import DBSCAN

from tools.silhouette import silhouette_score_anomaly

def find_best_eps(
    X,
    *args,
    min_eps=0.5,
    max_eps=1,
    delta=0.1,
    anomaly_markers=frozenset({-1}),
    handle_anomalies="singleton",
    pipeline=None,
    **kwargs,
):
    """Finds the best eps for a pipeline terminating in DBSCAN.
    Uses the silhouette score.

    Keyword arguments:
    X -- a feature matrix,
    *args -- args to be passed to sklearn.metrics.silhouette_score
    min_eps -- the minimum neighborhood size to try
    max_eps -- the maximum neighborhood size to try
    delta -- the change the the size of epsilon in the search
    anomaly_markers -- cluster labels that represent anomalys
    handle_anomalies -- how to handle anomalies must be 'singleton' or 'ignore'
    pipeline -- a sklearn.pipeline.Pipeline terminating in an instance of sklearn.cluster.DBSCAN
    **kwargs -- kwargs to be passed to sklearn.metrics.silhouette_score
    """

    if not pipeline:
        pipeline = make_pipeline(DBSCAN())

    best_eps = None
    best_score = -1

    for eps in np.arange(min_eps, max_eps, delta):

        pipeline[-1].set_params(eps=eps)

        labels = pipeline.fit(X)[-1].labels_

        try:
            score = silhouette_score_anomaly(
                X,
                labels,
                *args,
                anomaly_markers=anomaly_markers,
                handle_anomalies=handle_anomalies,
                **kwargs,
            )
        except ValueError as e:
            warnings.warn(f"When eps is {eps}, the following error occured: {e}\n")

        if score >= best_score:
            best_eps = eps
            best_score = score

    return best_eps, best_score