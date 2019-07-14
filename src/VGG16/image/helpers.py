import os
import numpy
from skimage import io, color, img_as_ubyte, measure
from operator import itemgetter
from PIL import Image


def get_image_entropies(dir_path):
    image_entropies = []
    counter = 0
    for filename in os.listdir(dir_path):
        if filename.endswith('.jpg'):
            path = dir_path + filename
            rgbImg = io.imread(path)
            grayImg = img_as_ubyte(color.rgb2gray(rgbImg))
            ent = measure.shannon_entropy(grayImg)
            dictionary = {"filename" : filename,
                          "entropy-value" : ent,
                          "counter" : counter}
            counter = counter + 1
            image_entropies.append(dictionary)

    image_entropies.sort(key=itemgetter('entropy-value'), reverse=True)
    print(image_entropies)
    return image_entropies


def crop_and_save_image(image_entropies, input_dir_path, output_dir_path):
    count = 0
    for dictionary in image_entropies[:10]:
        filename = dictionary["filename"]
        print(filename)
        rgb_img = io.imread(input_dir_path + filename)
        gray_img = img_as_ubyte(color.rgb2gray(rgb_img))

        img_to_crop = Image.fromarray(gray_img)
        # The crop rectangle, as a (left, upper, right, lower)-tuple.
        cropped_img = img_to_crop.crop((25, 20, 250, 275))
        cropped_img = numpy.array(cropped_img)
        count = count + 1
        print(output_dir_path + filename)
        io.imsave(output_dir_path + filename, cropped_img)
