from med2image import med2image


med2image.med2image_nii(
    inputFile="/Users/pmartyn/PycharmProjects/Thesis/Data/test/MRI-NII/T1/BP/CHRM2001_MPRAGE_T13D.nii",
    outputDir='/Users/pmartyn/PycharmProjects/Thesis/Data/test/tmp',
    outputFileStem='.jpg',
    outputFileType="jpg",
    # sliceToConvert="-1"

).run()


# med.run()