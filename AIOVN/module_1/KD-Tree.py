import math

class Node():
    def __init__(self, data, left=None, right=None):
        self.left = left
        self.right = right
        self.data = data

class Kd_tree():
    def build_kd_tree(self, node, depth=0):
        if not node:
            return None
        
        # assume all data have the same length
        k = len(node[0])
        # swap between x, y to compare for selecting node
        axis = depth % k
        # sort node base on axis 
        node.sort(key=lambda x: x[axis])
        # find median for parent node
        median = len(node) // 2
        # print
        print("data :", node[median])
        print("median :", median)
        print("------------------")
        return Node(
                data=node[median],
                left=self.build_kd_tree(node[:median], depth+1),
                right=self.build_kd_tree(node[median+1:], depth+1)
                )

    def distance_squared(self, point, compared):
        return math.sqrt(sum((x-y) ** 2 for x, y in zip(point, compared)))
    
    def closest_point(self, point, nearest_node, compared):
        # if compared is None then nearest_neighbor is also None
        if compared is None:
            return nearest_node
        if nearest_node is None:
            return compared
        if self.distance_squared(point, nearest_node) < self.distance_squared(point, compared):
            return nearest_node
        return compared

    def nearest_neighbor(self, point, node, depth=0, best=None):
        if point is None:
            return None
        # if retrived then have no branch or node to see then return the best with default is root node then (stored best compared node)
        if node is None:
            return best
        
        k = len(node.data)
        axis = depth % k

        if (point[axis] < node.data[axis]):
            next_branch = node.left
            opposite_branch = node.right
        else:
            next_branch = node.right
            opposite_branch = node.left

        best = self.closest_point(point, self.nearest_neighbor(point, next_branch, depth+1, best), node.data)

        if (point[axis] - node.data[axis]) ** 2 < self.distance_squared(point, best):
            best = self.closest_point(point, self.nearest_neighbor(point, opposite_branch, depth+1, best), best)

        return best

if __name__ == "__main__":
    points = [(7,2), (5,4), (9,6), (2,3), (4,7), (8,1)]
    _kd_tree = Kd_tree()
    _node = _kd_tree.build_kd_tree(points)
    query_point = (9,2)
    nearest_node = _kd_tree.nearest_neighbor(query_point, _node)
    print(nearest_node)


