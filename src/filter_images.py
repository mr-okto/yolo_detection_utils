import os
import cv2
import numpy
import argparse
from shutil import copy2

MIN_SIZE = 256


def filter_images():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--class", required=True,
        help="Class name of objects")
    ap.add_argument("-i", "--image_path", required=False,
        default="dataset",
        help="Path to label files")

    args = vars(ap.parse_args())
    obj_class = args['class']
    folder = '%s/%s' % (args['image_path'], obj_class)
    backup_folder = '%s_backup' % folder
    if not os.path.exists(backup_folder):
        os.mkdir(backup_folder)

    image_files = [f for f in os.listdir(folder) if not os.path.isdir(os.path.join(folder, f))]
    
    for f in image_files:
        # print('Processing file %s' % f)
        # check size of image and remove small ones
        file_path = os.path.join(folder, f)
        img = cv2.imread(file_path)
        (h, w) = img.shape[:2]

        # Copy file to backup folder
        copy2(file_path, backup_folder)
        
        if h < MIN_SIZE or w < MIN_SIZE:
            os.remove(file_path)


if __name__ == '__main__':
    filter_images()
