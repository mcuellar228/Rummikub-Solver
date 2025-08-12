import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=5):  # Reduced from 8 to 3
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_img = image.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def random_perspective_transform(image, max_warp=0.0):  # Reduced from 0.05 to 0.02
    rows, cols = image.shape[:2]

    def random_shift(max_shift):
        return np.random.uniform(-max_shift, max_shift)

    shift_x = max_warp * cols
    shift_y = max_warp * rows

    pts1 = np.float32([[0,0], [cols,0], [cols,rows], [0,rows]])
    pts2 = np.float32([
        [random_shift(shift_x), random_shift(shift_y)],
        [cols + random_shift(shift_x), random_shift(shift_y)],
        [cols + random_shift(shift_x), rows + random_shift(shift_y)],
        [random_shift(shift_x), rows + random_shift(shift_y)]
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    return warped

def augment_image(image, warp, scale, rotation, translation, sigma):
    rows, cols = image.shape[:2]

    # Rotation
    angle = np.random.uniform(-rotation, rotation)
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(image, M_rot, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

    # Translation
    tx = np.random.randint(-translation, translation)
    ty = np.random.randint(-translation, translation)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(rotated, M_trans, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

    scale = np.random.uniform(1-scale, 1+scale)
    resized = cv2.resize(translated, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # final = np.zeros_like(image)
    h, w = resized.shape[:2]
    top = max((rows - h) // 2, 0)
    bottom = max(rows - h - top, 0)
    left = max((cols - w) // 2, 0)
    right = max(cols - w - left, 0)

    final = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REPLICATE)
    h, w = resized.shape[:2]
    if h > rows:
        start_h = (h - rows) // 2
        resized = resized[start_h:start_h+rows, :]
    if w > cols:
        start_w = (w - cols) // 2
        resized = resized[:, start_w:start_w+cols]

    h, w = resized.shape[:2]
    y_offset = (rows - h) // 2
    x_offset = (cols - w) // 2
    final[y_offset:y_offset+h, x_offset:x_offset+w] = resized

    brightness = np.random.uniform(0.85, 1.15)
    final = np.clip(final * brightness, 0, 255).astype(np.uint8)
    final = np.clip(final, 0, 255).astype(np.uint8)

    final = random_perspective_transform(final, max_warp=warp)

    final = add_gaussian_noise(final, sigma=sigma)

    return final