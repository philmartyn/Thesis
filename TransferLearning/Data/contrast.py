import os

import cv2
base_path = '/Users/pmartyn/PycharmProjects/Thesis/TransferLearning/Data/All_Data_2/control/'
for filename in os.listdir(base_path):
    file = base_path + filename
    print(file)
    image = cv2.imread(file, 1)

    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))

    contrasted_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imwrite('/Users/pmartyn/PycharmProjects/Thesis/TransferLearning/Data/All_Data_2/control-contrast/' + filename, contrasted_image)

