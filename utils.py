import numpy as np

def get_union_box(boxes):
    """
    Calculates the union bounding box of all provided boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    if not boxes:
        return None
    
    boxes = np.array(boxes)
    x_min = np.min(boxes[:, 0])
    y_min = np.min(boxes[:, 1])
    x_max = np.max(boxes[:, 2])
    y_max = np.max(boxes[:, 3])
    
    return [x_min, y_min, x_max, y_max]

def get_padded_square_box(box, padding=0.2, img_shape=None):
    """
    Transforms a box into a square with padding.
    img_shape is (height, width)
    """
    x_min, y_min, x_max, y_max = box
    w = x_max - x_min
    h = y_max - y_min
    
    # Center of the original box
    cx = x_min + w / 2
    cy = y_min + h / 2
    
    # The side length of the square, including padding
    side = max(w, h) * (1 + padding)
    
    # Half-side for easy calculation
    hs = side / 2
    
    new_x1 = cx - hs
    new_y1 = cy - hs
    new_x2 = cx + hs
    new_y2 = cy + hs
    
    # If image shape is provided, try to keep the square within bounds
    if img_shape is not None:
        img_h, img_w = img_shape
        
        # Shift if out of bounds
        if new_x1 < 0:
            new_x2 -= new_x1
            new_x1 = 0
        if new_y1 < 0:
            new_y2 -= new_y1
            new_y1 = 0
        if new_x2 > img_w:
            new_x1 -= (new_x2 - img_w)
            new_x2 = img_w
        if new_y2 > img_h:
            new_y1 -= (new_y2 - img_h)
            new_y2 = img_h
            
        # Clamp again to be sure
        new_x1 = max(0, int(new_x1))
        new_y1 = max(0, int(new_y1))
        new_x2 = min(img_w, int(new_x2))
        new_y2 = min(img_h, int(new_y2))
    else:
        new_x1, new_y1, new_x2, new_y2 = int(new_x1), int(new_y1), int(new_x2), int(new_y2)
        
    return [new_x1, new_y1, new_x2, new_y2]

