import os
import cv2
import argparse
import numpy
import skimage
from skimage import io, color, img_as_ubyte
from operator import itemgetter
from PIL import Image
import random

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True, help="directory of images to process")
ap.add_argument("-o", "--output", required=True, help="output directory for processed images")
ap.add_argument("-t", "--type", required=True, help="either control or bipolar")
args = vars(ap.parse_args())

rotation_l = 15
rotation_r = 345

sample_type = args["type"]
dir_path = args["dir"]
output_dir = args["output"]

image_entropies = []
for filename in os.listdir(dir_path):
    if filename.endswith('.jpg'):
        path = dir_path + filename
        rgbImg = io.imread(path)
        grayImg = img_as_ubyte(color.rgb2gray(rgbImg))
        ent = skimage.measure.shannon_entropy(grayImg)
        dictionary= {"filename" : filename,
                     "entropy-value" : ent}
        image_entropies.append(dictionary)

image_entropies.sort(key = itemgetter('entropy-value'), reverse=True)

# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)

_, _, files = next(os.walk(output_dir))
file_count = len(files) + 1

count = 0
rnd_count = random.sample(range(file_count, file_count + 150), 150)


def save_image(file_counter, img):
    file_counter += 1
    img_name = sample_type + "-slice-" + str(rnd_count[count]) + ".jpg"
    print(output_dir + img_name)
    io.imsave(output_dir + img_name, img)
    return file_counter


def rotate(rotation, file_counter):
    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
    dst = cv2.warpAffine(cropped, m, (cols, rows))
    return save_image(file_counter, dst)


for dictionary in image_entropies[:50]:
    filename = dictionary["filename"]
    rgbImg = io.imread(dir_path + filename)
    grayImg = img_as_ubyte(color.rgb2gray(rgbImg))

    croppedImg = Image.fromarray(grayImg)
    cropped = croppedImg.crop((40,30,240,265))
    cropped = numpy.array(cropped)
    count = save_image(count, cropped)

    rows, cols = cropped.shape

    count = rotate(rotation_l, count)
    count = rotate(rotation_r, count)
