import numpy as np
import matplotlib.pyplot as plt
from tensorflow import mnist
import cv2
import csv

# Load data
(X_train_raw, y_train_raw), (_, _) = mnist.load_data()

def make_composite_digits(X_raw, y_raw, target_label, num_samples=1):
    composite_images = []
    composite_labels = []

    count = 0
    while count < num_samples:

        idx1 = np.random.randint(0, len(y_raw))
        idx2 = np.random.randint(0, len(y_raw))

        digit1, digit2 = y_raw[idx1], y_raw[idx2]

        if int(f"{digit1}{digit2}") == target_label:
            img1 = X_raw[idx1]
            img1 = img1[:, 2:22]
            img2 = X_raw[idx2]
            img2 = img2[:, 2:]

            combined = np.hstack((img1, img2))
            composite_images.append(combined)
            composite_labels.append(target_label)

            count += 1

    return np.array(composite_images), np.array(composite_labels)

for label in range(10, 14):
    images, labels = make_composite_digits(X_train_raw, y_train_raw, target_label=label, num_samples=1)
    plt.figure("Composite", figsize=(12, 6))
    for image in images:
        image = cv2.resize(image, (28, 28))
        image = image.flatten()
        file1_path = './Data/train2.csv'
        image = np.insert(image, 0, label)
        with open(file1_path, 'a', newline='') as file2:
            writer = csv.writer(file2)
            writer.writerow(image)

        # Uncomment below to see results
        # plt.imshow(image, cmap="gray")
        # plt.axis('off')
        # plt.show()



