import os
import argparse
import random
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True, help="directory of images to process")
ap.add_argument("-o", "--output", required=True, help="output directory for processed images")
ap.add_argument("-t", "--type", required=True, help="either control or bipolar")
args = vars(ap.parse_args())

dir_path = args["dir"]
output_dir = args["output"]
data_type = args["type"]

mylist = os.listdir(dir_path)

for x in range(0, 3):
    counter = 2
    random.shuffle(mylist)
    for filename in mylist[:640]:
        shutil.copy2(filename, "Crossfolds/Fold" + str(counter) + "/validation/" + data_type)
    for filename in mylist[-2272:]:
        shutil.copy2(filename, "Crossfolds/Fold" + str(counter) + "/train/" + data_type)
