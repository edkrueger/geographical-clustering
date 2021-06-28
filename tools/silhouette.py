from sklearn.metrics import silhouette_score

def _silhouette_score_anomaly_singleton(X, labels, *args, anomaly_markers, **kwargs):

    labels_relabeled = []
    marker = -1

    for label in labels:

        if label in anomaly_markers:
            labels_relabeled.append(marker)
            marker -= 1
        else:
            labels_relabeled.append(label)

    return silhouette_score(X, labels_relabeled, *args, **kwargs)

def _silhouette_score_anomaly_ignore(X, labels, *args, anomaly_markers, **kwargs):

    X_filtered = []
    labels_filtered = []

    for i in range(len(X)):

        if labels[i] not in anomaly_markers:
            X_filtered.append(X[i])
            labels_filtered.append(labels[i])

    return silhouette_score(X_filtered, labels_filtered, *args, **kwargs)

def silhouette_score_anomaly(
    X,
    labels,
    *args,
    anomaly_markers=frozenset({-1}),
    handle_anomalies="singleton",
    **kwargs
):

    if not handle_anomalies in {"ignore", "singleton"}:
        raise ValueError("handle_anomalies must take value 'ignore' or 'singleton'")

    if not anomaly_markers:
        return silhouette_score(X, labels, *args, **kwargs)

    if handle_anomalies == "ignore":
        return _silhouette_score_anomaly_ignore(
            X, labels, *args, anomaly_markers=anomaly_markers, **kwargs
        )

    if handle_anomalies == "singleton":
        return _silhouette_score_anomaly_singleton(
            X, labels, *args, anomaly_markers=anomaly_markers, **kwargs
        )

    raise ValueError("Something went wrong, check the arguments.")