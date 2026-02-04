class IoU:
    def __find_inter_point(self, boxA, boxB):
        # xA = max(boxA[0], boxB[0])
        # xB = min(boxA[2], boxB[2])
        # yA = max(boxA[1], boxB[1])
        # yB = min(boxA[3], boxB[3])
        A = (max(boxA[0], boxB[0]), max(boxA[1], boxB[1]))
        B = (min(boxA[2], boxB[2]), min(boxA[3], boxB[3]))
        return A, B

    def __compute_area_A(self, boxA):
        x1, y1, x2, y2 = boxA
        # add 1 for preventing two point have the same x or y location then make the value to be 0
        return (x2 - x1 + 1) * (y2 - y1 + 1)

    def __compute_area_B(self, boxB):
        x1, y1, x2, y2 = boxB
        # add 1 for preventing two point have the same x or y location then make the value to be 0
        return (x2 - x1 + 1) * (y2 - y1 + 1)

    def __compute_intersection(self, A, B):
        xA, yA = A
        xB, yB = B
        # add 1 for preventing two point have the same x or y location then make the value to be 0
        # use max to this make sure 2 area have no intersection with 0 value than minus value
        return max(0, xB - xA + 1) * max(0, yB - yA + 1)

    def __compute_union(self, area_A, area_B, intersection):
        return area_A + area_B - intersection

    def computeIoU(self, boxA, boxB):
        A, B = self.__find_inter_point(boxA, boxB)
        area_A = self.__compute_area_A(boxA)
        area_B = self.__compute_area_B(boxB)
        intersection = self.__compute_intersection(A, B)
        union = self.__compute_union(area_A, area_B, intersection)
        return float(intersection) / float(union)


if __name__ == "__main__":
    boxA = (0, 0, 5, 6)
    boxB = (7, 8, 10, 12)
    iou = IoU()
    print(iou.computeIoU(boxA=boxA, boxB=boxB))
