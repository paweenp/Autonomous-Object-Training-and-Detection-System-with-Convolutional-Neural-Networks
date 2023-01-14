import cv2
import numpy as np
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# Create different colors for each class.
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def draw_boxes(boxes, classes, labels, image):
    """
    Draws the bounding box around a detected object.
    """
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1) # Font thickness.
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            img=image,
            pt1=(int(box[0]), int(box[1])),
            pt2=(int(box[2]), int(box[3])),
            color=color[::-1], 
            thickness=lw
        )
        cv2.putText(
            img=image, 
            text=classes[i], 
            org=(int(box[0]), int(box[1]-5)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=lw / 3, 
            color=color[::-1], 
            thickness=tf, 
            lineType=cv2.LINE_AA
        )
    return image