import numpy as np
import pandas as pd

def cals_map_all(test_loaders, model, device):
    model = model.eval()
    metr = []
    for loader in test_loaders:
        e_list, l_list = encode(model, loader, device)
        embs = torch.cat(e_list,dim=0).numpy()
        lbs = torch.cat(l_list,dim=0).numpy()
        metr.append(calc_map(embs, lbs))
    return np.mean(metr)

def calc_map(X, labels, K = 5):
    metr = []
    for i in range(len(X)):
        ind = X[i]
        label = labels[i]
        is_valid_label = (label[ind[:,1:]] == label.reshape(-1,1)).astype(int)

        cum_sum = np.cumsum(is_valid_label, axis=1)
        P_K = cum_sum/np.arange(1, K+1).reshape(1,-1)
        AP_K = P_K.sum(axis=1) / np.clip(cum_sum[:,-1],1, K)
        metr.append(AP_K.mean())

    return np.mean(metr)

def get_nearest_idxs(data):
    matrix = data
    labels = list(range(len(data)))

    labeled_matrix = []
    for i in range(len(matrix)):
        distances = {labels[j]: matrix[i][j] for j in range(len(matrix))}
        labeled_matrix.append({labels[i]: distances})

    for i, elem in enumerate(labeled_matrix):
        labeled_matrix[i] = {label: sorted(dist.items(), reverse=True, key=lambda x: x[1]) for label, dist in elem.items()}

    closest_labels_matrix = []
    closest_distances_matrix = []
    for labeled_distances in labeled_matrix:
        for label, distances in labeled_distances.items():
            closest_labels = [x[0] for x in distances[:6]]
            closest_distances = [x[1] for x in distances[:6]]
            closest_labels_matrix.append(closest_labels)
            closest_distances_matrix.append(closest_distances)

    return np.array(closest_labels_matrix)