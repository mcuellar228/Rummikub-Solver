import cv2
import os

for file in os.listdir("Data/IndividualTiles/Raw/"):
    full_path = os.path.join("Data/IndividualTiles/Raw/", file)
    file_name = file.split('.')[0]
    image = cv2.imread(full_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 150, 256, cv2.THRESH_BINARY)
    thresh2 = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,10)

    resizedThresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_NEAREST)
    resizedThresh2 = cv2.resize(thresh2, (32, 32), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(f"Data/IndividualTiles/ThresholdedAndResized/{file_name}_thresh.png", resizedThresh)
    cv2.imwrite(f"Data/IndividualTiles/ThresholdedAndResized/{file_name}_thresh2.png", resizedThresh2)
