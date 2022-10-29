from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import numpy as np


def hungray_aligment(y_true, y_pred):
    """
    Align cluster assignments and fine-grained labels with Hungarian algorithm
    for accuracy evaluation.
    """
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    """
    Calculate clustering accuracy.
    """
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def clustering_score(y_true, y_pred):
    """
    Calculate evaluation metrics: ACC (Clustering Accuracy), ARI (Adjusted Rand Score) and 
    NMI (Normalized Mutual Information).
    """
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2)}