import os
import argparse
import numpy
from skimage import io, color, img_as_ubyte
from PIL import Image
import random
from src.predictor.image import helpers

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

image_entropies = helpers.get_image_entropies(dir_path)

# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)

_, _, files = next(os.walk(output_dir))
file_count = len(files) + 1

count = 0
rnd_count = random.sample(range(file_count, file_count + 32), 32)


def save_image(file_counter, img):
    file_counter += 1
    img_name = sample_type + "-slice-" + str(rnd_count[count]) + ".jpg"
    print(output_dir + img_name)
    io.imsave(output_dir + img_name, img)
    return file_counter


for dictionary in image_entropies[:32]:
    filename = dictionary["filename"]
    rgbImg = io.imread(dir_path + filename)
    grayImg = img_as_ubyte(color.rgb2gray(rgbImg))

    croppedImg = Image.fromarray(grayImg)
    # The crop rectangle, as a (left, upper, right, lower)-tuple.
    cropped = croppedImg.crop((25,20,250,275))
    cropped = numpy.array(cropped)
    count = save_image(count, cropped)
