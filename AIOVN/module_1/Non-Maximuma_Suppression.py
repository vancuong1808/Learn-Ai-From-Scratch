from IoU import IoU


class NNM:
    def non_max_suppression(self, scores, boxes, iou_threshold):
        sorted_boxes = [
            box
            for box, _ in sorted(zip(boxes, scores), key=lambda x: x[1], reverse=True)
        ]
        kept_box = []
        filtered_box = []
        iou = IoU()

        while sorted_boxes:
            highest = sorted_boxes.pop(0)
            kept_box.append(highest)
            for box in sorted_boxes:
                iou_compared = iou.computeIoU(highest, box)
                # check the iou values similar to the highest to remove make sure have only the best bounding box
                # remove the bounding box detect the same object with the highest
                if iou_compared <= iou_threshold:
                    # add the bounding box detect another object
                    filtered_box.append(box)
            # after done have filtered_box by using the highest bound box score then loop and find another box with the high score
            sorted_boxes = filtered_box

        return kept_box


if __name__ == "__main__":
    boxes = [
        (12, 84, 140, 212),
        (24, 84, 152, 212),
        (36, 84, 164, 212),
        (12, 96, 140, 224),
        (24, 96, 152, 224),
        (24, 108, 152, 236),
    ]
    scores = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    iou_threshold = 0.3
    nnm = NNM()
    print(nnm.non_max_suppression(scores, boxes, iou_threshold))
