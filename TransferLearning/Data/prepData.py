from PIL import Image
from deepbrain import Extractor
import nibabel as nib
import os
import numpy as np
from operator import itemgetter
from skimage import io, color, measure

# Taken from https://github.com/iitzco/deepbrain/blob/master/bin/deepbrain-extractor
# Changed the code a little because I didn't want the brain NII files saved to disk.

# This is not being used as it produces images that are not predicted correctly
# while the prepDataIO produces images which do predict correctly.
# I'm not sure why this doesn't work.


def prep_data(filename_path):

    nii_output_dir = os.getenv('TMP_DIR_PATH', "/Users/pmartyn/PycharmProjects/Thesis/tmp/")

    if not os.path.exists(nii_output_dir):
        os.makedirs(nii_output_dir)

    p = 0.5
    img = nib.load(filename_path)

    affine = img.affine
    img = img.get_fdata()

    prob = Extractor().run(img)
    mask = prob > p

    brain = img[:]
    brain[~mask] = 0
    image_array = nib.Nifti1Image(brain, affine).get_fdata()

    total_slices = image_array.shape[2]

    image_entropies = []

    for current_slice in range(0, total_slices):
        image_data = np.rot90(np.rot90(np.rot90(image_array[:, :, current_slice])))
        gray_image_data = color.rgb2gray(image_data)

        ent = measure.shannon_entropy(gray_image_data)
        dictionary = {"filename": current_slice,
                      "entropy-value": ent,
                      "image-data": gray_image_data}

        image_entropies.append(dictionary)

    image_entropies.sort(key=itemgetter('entropy-value'), reverse=True)

    for dictionary in image_entropies[:6]:
        filename = dictionary["filename"]
        image_data = dictionary["image-data"]
        img_to_crop = Image.fromarray(image_data)
        # The crop rectangle, as a (left, upper, right, lower)-tuple.
        cropped = img_to_crop.crop((25, 20, 250, 275))
        cropped = np.array(cropped)
        io.imsave(nii_output_dir + str(filename) + ".jpeg", cropped)
