from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import heapq
from annoy import AnnoyIndex

#Annoy
class Node:
    def __init__(self, point, index=None, left=None, right=None):
        self.point = point
        self.index = index
        self.left = left
        self.right = right

class AnnoyTree:
    def __init__(self, data, n_trees=10, search_k=-1):
        self.data = data
        self.n_trees = n_trees
        self.search_k = search_k if search_k != -1 else n_trees * data.shape[0] // 10
        self.trees = [self._build_tree() for _ in range(n_trees)]
    
    def _build_tree(self):
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)
        return self._make_tree(indices)
    
    def _make_tree(self, indices):
        if len(indices) <= 1:
            return {'index': indices[0]} if len(indices) == 1 else None
        
        # Выбор случайного вектора для разделения выборки
        split_index = indices[np.random.randint(len(indices))]
        split_vector = self.data[split_index]
        distances = cosine_similarity(self.data[indices], split_vector.reshape(1, -1)).flatten()
        median_val = np.median(distances)
        
        left_indices = indices[distances < median_val]
        right_indices = indices[distances >= median_val]
        
        # Предотвращение бесконечной рекурсии пополам
        if len(left_indices) == 0 or len(right_indices) == 0:
            left_indices = indices[:len(indices) // 2]
            right_indices = indices[len(indices) // 2:]
        
        return {
            'index': split_index,
            'left': self._make_tree(left_indices),
            'right': self._make_tree(right_indices)
        }
    
    def _search_tree(self, node, target_vector, k, heap):
        if node is None or isinstance(node, int):
            return

        if 'index' in node:
            dist = 1 - cosine_similarity([self.data[node['index']]], [target_vector])[0, 0]
            if len(heap) < k or dist < -heap[0][0]:
                if len(heap) == k:
                    heapq.heappop(heap)
                heapq.heappush(heap, (-dist, node['index']))

        # Поиск с приоритетом более близкой стороны
        split_vector = self.data[node['index']]
        dist_to_split = 1 - cosine_similarity([split_vector], [target_vector])[0, 0]
        go_right = dist_to_split < 0
        first_side = 'right' if go_right else 'left'
        second_side = 'left' if go_right else 'right'

        self._search_tree(node.get(first_side), target_vector, k, heap)
        if len(heap) < k or abs(dist_to_split) < -heap[0][0]:
            self._search_tree(node.get(second_side), target_vector, k, heap)

    def search(self, query_points, k=1):
        results = []
        for query_point in query_points:
            heap = []
            for tree in self.trees:
                self._search_tree(tree, query_point, k, heap)
            heap.sort(reverse=True)
            indices = [index for _, index in heap]
            results.append(indices[:k])
        return results

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
        
# class KDTree:
#     def __init__(self, data, depth=0):
#         points = list(enumerate(data))
#         self.root = self._build_tree(points)
#         self.depth = depth

#     def _build_tree(self, points, depth=0):    
#         if not points:
#             return None

#         k = len(points[0][1])  
#         axis = depth % k
#         points.sort(key=lambda x: x[1][axis]) 
#         median = len(points) // 2 

#         return Node(
#             point = points[median][1],
#             value = points[median][1], 
#             index = points[median][0], 
#             left = self._build_tree(points[:median], depth + 1),
#             right = self._build_tree(points[median + 1:], depth + 1)
#         )

#     def _cosine_distance(self, x, y):
#         return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
  
#     def _find_nn(self, node, point, depth=0, best=None):
#         if node is None:
#             return best

#         axis = depth % len(point)
#         next_branch = node.left if point[axis] < node.point[axis] else node.right
#         opposite_branch = node.right if next_branch == node.left else node.left
       
#         best_next = self._find_nn(next_branch, point, depth + 1, best)  
        
#         if best_next is None:  
#             best_next = []

#         if len(best_next) < 6 or self._cosine_distance(point, node.value) < -best_next[0][0]:
#             heapq.heappush(best_next, (-self._cosine_distance(point, node.value), node))
#             if len(best_next) > 6:  
#                 heapq.heappop(best_next) 
        
#         best = best_next
  
#         if opposite_branch is not None and (len(best) < 6 or abs(node.point[axis] - point[axis]) < -best[0][0]):
#             best = self._find_nn(opposite_branch, point, depth + 1, best)
        
#         return best

#     def search(self, data):
#         points_to_search = list(data)
#         output = []
#         for idx, point in enumerate(points_to_search):   
#             nearest_neighbors = self._find_nn(self.root, point, 0)
#             nearest_neighbors.sort(reverse=True)
#             indices = [nn.index for sim, nn in nearest_neighbors] 
#             output.append(indices)
#         return np.array(output)

# Создаем экземпляров класса и используем его:
# tree = KDTree(data_norm)
# result = tree.search(data_norm)
# print(result)

#  points = list(enumerate(data_norm))
#  root = kd_tree(points)

#  points_to_search = list(data_norm)

#  output = []

#  ищем наиболее похожие векторы для каждого вектора
#     for idx, point in enumerate(points_to_search):
#         nearest_neighbors = []
#         nearest_neighbors = find_nn(root, point, 0, nearest_neighbors)
#         nearest_neighbors.sort(reverse=True)
#         #берем только индексы ближайших соседей, но не самого вектора
#         indices = [nn.index for sim, nn in nearest_neighbors]
#         output.append(indices)

#     print(np.array(output))

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