import os
import numpy
from skimage import io, color, img_as_ubyte, measure
from operator import itemgetter
from PIL import Image


def get_image_entropies(dir_path):
    image_entropies = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.jpg'):
            path = dir_path + filename
            rgbImg = io.imread(path)
            grayImg = img_as_ubyte(color.rgb2gray(rgbImg))
            ent = measure.shannon_entropy(grayImg)
            dictionary = {"filename" : filename,
                          "entropy-value" : ent}
            image_entropies.append(dictionary)

    image_entropies.sort(key=itemgetter('entropy-value'), reverse=True)
    return image_entropies


def crop_image(image_entropies, input_dir_path, output_dir_path):
    count = 0
    for dictionary in image_entropies[:6]:
        filename = dictionary["filename"]
        print(filename)
        rgbImg = io.imread(input_dir_path + filename)
        grayImg = img_as_ubyte(color.rgb2gray(rgbImg))

        croppedImg = Image.fromarray(grayImg)
        # FIXME : Crop less at the bottom, more at the top
        # TODO : Resize to the same as the ALZ data
        # The crop rectangle, as a (left, upper, right, lower)-tuple.
        cropped = croppedImg.crop((25, 20, 250, 275))
        cropped = numpy.array(cropped)
        count = count + 1
        print(output_dir_path + str(count) + ".jpg")
        io.imsave(output_dir_path + str(count) + ".jpg", cropped)
