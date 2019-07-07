import imageio
from scipy import misc
from med2image import med2image
from deepbrain import Extractor
import nibabel as nib
import os
from Data import helpers, NII_to_JPG
from PIL import Image
import numpy as np

# Taken from https://github.com/iitzco/deepbrain/blob/master/bin/deepbrain-extractor
# Changed the code a little because I didn't want the brain mask image saved.


def prep_data(filename):

    input_file = "/Users/pmartyn/Documents/College_Work/Thesis/Thesis-Frontend/thesis-frontend/resources/public/css/" + filename
    nii_output_dir = "/Users/pmartyn/PycharmProjects/Thesis/tmp"

    if not os.path.exists(nii_output_dir):
        os.makedirs(nii_output_dir)

    p = 0.5
    img = nib.load(input_file)
    print("img: ", type(img))

    affine = img.affine
    img = img.get_fdata()

    extractor = Extractor()

    prob = extractor.run(img)
    mask = prob > p


    brain = img[:]
    brain[~mask] = 0
    brain = nib.Nifti1Image(brain, affine).get_fdata()
    image_array = brain
    print("Brain shape : ", image_array.shape[2])

    total_slices = image_array.shape[2]
    slice_counter = 0

    images = []
    data = np.rot90(np.rot90(np.rot90(image_array[:, :, 100])))
    imageio.imwrite("/Users/pmartyn/PycharmProjects/Thesis/Data/MRI-NII/tmp/output-on/test.png", data)
    # for current_slice in range(0, total_slices):
    #     # alternate slices
    #     if (slice_counter % 1) == 0:
    #         data = image_array[:, :, current_slice]
    #
    #         # alternate slices and save as png
    #         if (slice_counter % 1) == 0:
    #             print('Saving image...')
    #             images.append(data)
    #
    #             # image_name = inputfile[:-4] + "_z" + "{:0>3}".format(str(current_slice + 1)) + ".png"
    #             # misc.imsave(image_name, data)
    #             print('Saved.')
    #
    #
    #             slice_counter += 1
    #             print('Moved.')
    #
    # print('Finished converting images')


    # image_array = brain.get_data()
    # print("Brian image array: ", type(image_array))
    # print(image_array)
    # img = Image.fromarray(a[90], 'RGB')
    # img.show()

    # NII_to_JPG.splittNII(brain)
    # nib.save(brain, os.path.join(nii_output_dir, "brain.nii"))
    #
    # jpg_output_dir_path = nii_output_dir + '/jpg/'
    #
    # if not os.path.exists(jpg_output_dir_path):
    #     os.makedirs(jpg_output_dir_path)
    #
    # med2image.med2image_nii(
    #     inputFile=nii_output_dir + "/brain.nii",
    #     outputDir=jpg_output_dir_path,
    #     outputFileStem='output-file',
    #     outputFileType="jpg").run()
    #
    # prepped_img_dir = nii_output_dir + "/proc_img/"
    #
    # if not os.path.exists(prepped_img_dir):
    #     os.makedirs(prepped_img_dir)
    #
    image_entropies = helpers.get_image_entropies(jpg_output_dir_path)
    helpers.crop_image(image_entropies, jpg_output_dir_path, prepped_img_dir)


prep_data("sub-60005_anat_sub-60005_T1w.nii")
