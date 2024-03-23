from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import heapq
from annoy import AnnoyIndex

#KDTree
class Node:
    def __init__(self, point, left=None, right=None, value=None, index=None):
        self.point = point
        self.left = left
        self.right = right
        self.value = value
        self.index = index

class KDTree:
    class Node:
        def __init__(self, point, value, index, left=None, right=None):
            self.point = point
            self.value = value
            self.index = index
            self.left = left
            self.right = right

    def __init__(self, data, depth=0):
        points = list(enumerate(data))
        self.root = self._build_tree(points)
        self.depth = depth

    def _build_tree(self, points, depth=0):    
        if not points:
            return None

        k = len(points[0][1])  
        axis = depth % k
        points.sort(key=lambda x: x[1][axis]) 
        median = len(points) // 2 

        return Node(
            point = points[median][1],
            value = points[median][1], 
            index = points[median][0], 
            left = self._build_tree(points[:median], depth + 1),
            right = self._build_tree(points[median + 1:], depth + 1)
        )

    def _cosine_distance(self, point1, point2):
        return 1 - np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))

    def _build_tree(self, points, depth=0):
        if not points:
            return None
        k = len(points[0][1])
        axis = depth % k
        points.sort(key=lambda x: x[1][axis])
        median = len(points) // 2
        return self.Node(
            point=points[median][1],
            value=points[median][1],
            index=points[median][0],
            left=self._build_tree(points[:median], depth + 1),
            right=self._build_tree(points[median + 1:], depth + 1)
        )

    def _find_nn(self, node, point, depth=0, best=None):
        if node is None:
            return best
        if best is None:
            best = []
        axis = depth % len(point)
        
        next_branch = None
        opposite_branch = None
        if point[axis] < node.point[axis]:
            next_branch = node.left
            opposite_branch = node.right
        else:
            next_branch = node.right
            opposite_branch = node.left
        
        self._find_nn(next_branch, point, depth + 1, best)
        
        dist = self._cosine_distance(point, node.value)
        
        if len(best) < 6:
            heapq.heappush(best, (-dist, node.index))
        else:
            heapq.heappushpop(best, (-dist, node.index))
        
        if opposite_branch is not None:
            if len(best) < 6 or abs(point[axis] - node.point[axis]) < -best[0][0]:
                self._find_nn(opposite_branch, point, depth + 1, best)
        
        return best

    def search(self, data, k=6):
        result = []
        for point in data:
            nearest = self._find_nn(self.root, point, 0)
            nearest.sort(reverse=True)
            nearest_indices = [index for dist, index in nearest]
            result.append(nearest_indices[:k])
        return np.array(result)     

#KNN
class DistributedCosineKnn:
    def __init__(self, k=5):
        self.k = k

    def fit(self, input_data, n_bucket=1):
        idxs = []
        dists = []
        buckets = np.array_split(input_data, n_bucket)
        for b in range(n_bucket):
            cosim = cosine_similarity(buckets[b], input_data)
            idx0 = [(heapq.nlargest((self.k + 1), range(len(i)), i.take)) for i in cosim]
            idxs.extend(idx0)
            dists.extend([cosim[i][idx0[i]] for i in range(len(cosim))])
        return np.array(idxs), np.array(dists)

#HNSW
class HNSW:
    def __init__(self, X):
        self.num_vectors = len(X)
        self.num_axes = len(X[0])
        self.index = AnnoyIndex(self.num_axes, 'angular')

    def build_index(self, vectors):
        for i, vector in enumerate(vectors):
            self.index.add_item(i, vector)
        self.index.build(10)

    def search(self, X, points_to_search, num_neighbors=6):
        self.build_index(X)
        result = []
        for point in points_to_search:
            nn = self.index.get_nns_by_vector(point, num_neighbors)
            result.append(nn)
        return np.array(result)