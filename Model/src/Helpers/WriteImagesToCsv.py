import os
import sys
import cv2
import csv
import numpy as np
DIRECTORY = os.environ.get('DIRECTORY')

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from Model.src.ComputerVision.AugmentImages import augment_image


filename = "train20.csv"

label_count = {}

with open(filename, 'w', newline='') as csvfile:
  csv_writer = csv.writer(csvfile)

  for file in os.listdir(DIRECTORY):
    full_path = os.path.join(DIRECTORY, file)
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    label = file.split('.')[0].split('_')[1]
    label = 0 if label == 'Joker' else int(label)

    if label_count.get(label, 0) >= 4000:
      continue

    for i in range(500):

      if label_count.get(label, 0) >= 4000:
        break

      augmented_img = augment_image(image, 0.01, 0.1, 3, 2, 2)
      augmented_img2 = augment_image(image, 0.1, 0.3, 15, 6, 8)

      _, thresh = cv2.threshold(augmented_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      _, thresh2 = cv2.threshold(augmented_img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

      resizedThresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_NEAREST)
      resizedThresh2 = cv2.resize(thresh2, (32, 32), interpolation=cv2.INTER_NEAREST)

      image_data = resizedThresh.flatten()
      image_data = image_data.tolist()

      image_data2 = resizedThresh2.flatten()
      image_data2 = image_data2.tolist()

      row_data = [label] + image_data
      row_data2 = [label] + image_data2

      label_count[label] = label_count.get(label, 0) + 2

      csv_writer.writerow(row_data)
      csv_writer.writerow(row_data2)

      # Uncomment below to see augmented and thresholded images
      # cv2.imshow('Augmented Tile', resizedThresh)
      # cv2.moveWindow('Augmented Tile', 100, 100)  # Move to (x=100, y=100)

      # cv2.imshow('Augmented Tile2', resizedThresh2)
      # cv2.moveWindow('Augmented Tile2', 300, 100)  # Move second window to avoid overlap

      # cv2.waitKey(0)
      # cv2.destroyAllWindows()