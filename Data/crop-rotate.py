import os
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="image to process")
ap.add_argument("-t", "--type", required=True, help="either control or bipolar")

args = vars(ap.parse_args())

image_path = args["image"]
type = args["type"]
rotation_l = 15
rotation_r = 345

path, dirs, files = next(os.walk(image_path))
file_count = len(files)
counter = 0

for filename in os.listdir(image_path):
    file_count = counter
    img = cv2.imread(filename, 0)

    rows, cols = img.shape

    M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_l, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))

    new_count = file_count + 1
    print(image_path + "/" + type + "-slice" + str(new_count) + ".jpg")
    cv2.imwrite(image_path + "/" + type + "-slice" + str(new_count) + ".jpg", dst)

    M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_r, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))

    file_count = len(files)

    new_count = file_count + 2
    print(image_path + "/" + type + "-slice" + str(new_count) + ".jpg")
    cv2.imwrite(image_path + "/" + type + "-slice" + str(new_count) + ".jpg", dst)
    counter = new_count

