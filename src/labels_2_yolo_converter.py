# -*- coding: utf-8 -*-

import os
import sys
import cv2
import json
import argparse
from os import walk, getcwd


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_labels():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--class", required=True,
        help="Class name of objects")
    ap.add_argument("-f", "--names_file", required=True,
        help="File with class names")
    ap.add_argument("-l", "--label_path", required=False,
        default="./dataset/labels",
        help="Path to label files")
    ap.add_argument("-i", "--image_path", required=False,
        default="./dataset",
        help="Path to label files")
    ap.add_argument("-o", "--out_path", required=False,
        default="./result",
        help="Path to output txt files")

    args = vars(ap.parse_args())

    classes = []
    with open(args['names_file']) as f:
        for line in f:
            class_name = line.rstrip('\n')
            print(class_name)
            classes.append(class_name)

    """ Configure Paths"""
    class_name = args['class']
    class_num = classes.index(class_name) + 1
    # print(class_num)
    mypath = "%s/%s/" % (args['label_path'], class_name)
    outpath = "%s/%s/" % (args['out_path'], class_name)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    list_file = open('%s_list.txt' % (class_name), 'w')

    """ Get input json file list """
    json_name_list = []
    for file in os.listdir(mypath):
        if file.endswith(".json"):
            json_name_list.append(file)

    print(len(json_name_list))

    for json_name in json_name_list:
        txt_name = json_name.rstrip(".json") + ".txt"
        file_path = mypath + json_name
        print("Input:" + file_path)

        txt_outpath = outpath + txt_name
        print("Output:" + txt_outpath)
        txt_outfile = open(txt_outpath, "a")

        img_path = str('%s/%s/%s.jpg' % (args['image_path'], class_name, os.path.splitext(json_name)[0]))
        if not os.path.exists(img_path):
            os.remove(file_path)
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        with open(file_path) as f:
            data = json.load(f)
            shapes = data['shapes']
            processed_shapes = []
            for shape in shapes:
                if shapes in processed_shapes:
                    continue
                points = shape['points']
                x1 = float(points[1][0])
                y1 = float(points[1][1])
                x2 = float(points[0][0])
                y2 = float(points[0][1])

                # exclude bbox with negative points
                if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                    print('Broken bbox found\n')
                    continue

                # in case when labelling, points are not in the right order
                xmin = min(x1, x2)
                xmax = max(x1, x2)
                ymin = min(y1, y2)
                ymax = max(y1, y2)

                # print(xmin, xmax, ymin, ymax)
                b = (xmin, xmax, ymin, ymax)
                bb = convert((w, h), b)
                # bb = (x, y, w, h)
                # print(bb)
                txt_outfile.write("%d %s\n" % (class_num, " ".join([str(a) for a in bb])))
                processed_shapes.append(shape)
        txt_outfile.close()
        out_file_size = int(os.stat(txt_outpath).st_size)
        
        if out_file_size == 0:
            print('Empty file')
            os.remove(txt_outpath)
        else:
            """ Save those images with bb into list"""
            list_file.write('%s/%s/%s.jpg\n' % (args['image_path'], class_name, os.path.splitext(txt_name)[0]))

    list_file.close()


if __name__ == '__main__':
    convert_labels()
