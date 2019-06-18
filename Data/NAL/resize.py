import os
import numpy
from skimage import io
from PIL import Image

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dir", required=False, help="directory of images to process")
# ap.add_argument("-o", "--output", required=False, help="output directory for processed images")
# ap.add_argument("-t", "--type", required=False, help="either control or bipolar")
# args = vars(ap.parse_args())

sample_type = "control"
dir_path = '/Users/pmartyn/PycharmProjects/Thesis/Data/NAL/Val/'
output_dir = '/Users/pmartyn/PycharmProjects/Thesis/Data/NAL/Output/'

count = 321


def save_image(file_counter, img):
    img_name = sample_type + "-slice-" + str(file_counter) + ".jpg"
    print(output_dir + img_name)
    io.imsave(output_dir + img_name, img)
    return file_counter + 1


for filename in os.listdir(dir_path):
    if filename.endswith('.jpg'):
        path = dir_path + filename
        img = io.imread(path)
        img = Image.fromarray(img)
        baseheight = 255
        hpercent = (baseheight / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        resized_img = img.resize((wsize, baseheight), Image.ANTIALIAS)

        resized_img = numpy.array(resized_img)
        count = save_image(count, resized_img)

# 2272 Train Bipolar
# 1024 Train Control - 1248 NAL
# 640 Val Bipolar
# 320 Val Control - 320 NAL
