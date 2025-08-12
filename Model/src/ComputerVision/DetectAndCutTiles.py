import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from Model_1024 import make_predictions

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def isCircleish(contour, tolerance=0.3):
    area = cv2.contourArea(contour)
    _, _, radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * radius * radius
    area_ratio = area / circle_area
    return abs(area_ratio - 1.0) < tolerance

def isSkinnyShape(contour, aspect_threshold=3):
    _, _, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return False
    aspect_ratio = max(w, h) / min(w, h)
    if aspect_ratio >= aspect_threshold:
        return True
    return False

def thresholdAndResize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resizedThresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_NEAREST)
    resizedThresh = resizedThresh/255
    return resizedThresh.flatten().reshape(1024, 1), resizedThresh

def detectColor(image, mask):
    masked_pixels = image[mask > 0]
    if masked_pixels.size == 0:
        return "unknown"

    hsv_pixels = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
    h, s, v = hsv_pixels[:, 0, 0], hsv_pixels[:, 0, 1], hsv_pixels[:, 0, 2]

    color_ranges = {
        'black':  ((s < 60) & (v < 60)),
        'blue':   ((h >= 90) & (h <= 130) & (s > 60) & (v > 60)),
        'red':    (((h <= 8) | (h >= 172)) & (s > 120) & (v > 60)),
        'orange': ((h >= 12) & (h <= 20) & (s > 130) & (v > 60)),
        'cream':  ((s < 40) & (v >= 200))
    }

    light_red_mask = (((h <= 10) | (h >= 170)) & (s > 50) & (s <= 120) & (v > 100))
    light_orange_mask = ((h >= 10) & (h <= 25) & (s > 60) & (s <= 130) & (v > 100))

    for color, rule in color_ranges.items():
        if np.any(rule):
            return color

    if np.any(light_red_mask):
        return 'red'
    if np.any(light_orange_mask):
        return 'orange'

    cream_mask = (s <= 90) & (v >= 180)
    if np.all(cream_mask):
        return 'cream'

    return 'other'

def padAndCropImage(contour, image, output_dir, count, color):
    valid_contour_count = count
    x, y, w, h = cv2.boundingRect(contour)

    padding = 30
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    cropped = image[y:y+h, x:x+w]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    model_input, resizedThresh = thresholdAndResize(cropped)
    prediction, A4 = (make_predictions(model_input))
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cropped_rgb)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(resizedThresh, cmap='gray')
    plt.title("Processed for model")
    plt.axis('off')

    plt.subplots_adjust(bottom=0.2)
    plt.figtext(0.5, 0.1, f"Prediction: {color} {prediction[0]}", ha='center', fontsize=14)
    plt.show()

    output_path = os.path.join(output_dir, f"tile_{valid_contour_count:03d}.png")

    valid_contour_count+=1
    print(f"Saved tile {valid_contour_count}: {output_path}")

def predictNumber(contour, image):
    x, y, w, h = cv2.boundingRect(contour)

    padding = 30
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    cropped = image[y:y+h, x:x+w]
    model_input, resizedThresh = thresholdAndResize(cropped)
    prediction, A4 = (make_predictions(model_input))
    predicted_number = prediction[0].item()
    return predicted_number

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def get_closed_edges(image, kernel_small_size, kernel_large_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    median_brightness = np.median(gray)
    if median_brightness < 100:
        beta = 100-median_brightness
    else:
        beta = 0
    alpha = 1.5

    enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    blurred1 = cv2.bilateralFilter(gray, 15, 200, 200)
    blurred2 = cv2.GaussianBlur(enhanced, (7,7), 30)
    blurred = cv2.bitwise_and(blurred1,blurred2)

    edges = cv2.Canny(blurred, threshold1=10, threshold2=35)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_small_size, kernel_small_size))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_large_size, kernel_large_size))
    result_small = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_small)
    result_large = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_large)
    closed_edges = cv2.bitwise_and(result_small, result_large)

    return closed_edges

def is_two_levels_deep(hierarchy, idx):
    parent = hierarchy[0][idx][3]
    if parent == -1:
        return False
    grandparent = hierarchy[0][parent][3]
    if grandparent == -1:
        return False
    great_grandparent = hierarchy[0][grandparent][3]
    return great_grandparent == -1

def detectTiles(file_storage):
    update_kernel = False
    tiles = []
    file_bytes = file_storage.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    closed_edges = get_closed_edges(image, 17, 50)
    contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result_image = image.copy()

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 2000 and is_two_levels_deep(hierarchy, i):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            color = detectColor(image, mask)
            if not color == 'cream' and not color == 'other' and not isSkinnyShape(contour):
                x, y, w, h = cv2.boundingRect(contour)
                if h > 120:
                    update_kernel = True
                    break
                cv2.drawContours(result_image, [contour], -1, (0,100,0), 6)
                number = predictNumber(contour, image)
                tiles.append({"number":number,"color":color})
                # padAndCropImage(contour, image, "./CroppedTiles/", i, color)
    if update_kernel:

        closed_edges = get_closed_edges(image, 50, 100)
        contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        result_image = image.copy()

        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 2000 and is_two_levels_deep(hierarchy, i):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                color = detectColor(image, mask)
                if not color == 'cream' and not color == 'other' and not isSkinnyShape(contour):
                    cv2.drawContours(result_image, [contour], -1, (0,100,0), 6)
                    number = predictNumber(contour, image)
                    tiles.append({"number":number,"color":color})
                    # padAndCropImage(contour, image, "./CroppedTiles/", i, color)

    # uncomment to save canny edge and contour images
    # cv2.imwrite("closed_canny_edges.png", closed_edges)
    # cv2.imwrite("find_contours.png", result_image)
    return tiles