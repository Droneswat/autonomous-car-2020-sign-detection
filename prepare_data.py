from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import pandas as pd 
from tqdm import tqdm
import numpy as np
import glob
import json
import cv2
import shutil


def display_sample(annotation_dict):
    """
    Display the image with bounding box
    """
    for abs_img_path, annotation_list in annotation_dict.items():
        img = cv2.imread(abs_img_path)
        for annotation in annotation_list:
            x1 = annotation['x1']
            y1 = annotation['y1']
            x2 = annotation['x2']
            y2 = annotation['y2']
            idx = annotation['idx']
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, idx*255, (1-idx)*255), 2)
            text = 'left'
            if idx == 1:
                text = 'right'
            cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)

        cv2.namedWindow("Sample", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Sample",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Sample", img)

        k = cv2.waitKey(0)
        if k == 110:
            continue
        else:
            break

    cv2.destroyAllWindows()


def write_new_annotation(annotation_dict, out_annotation_file):
    """
    Write data from annotation dict to a new annotation file
    """
    with open(out_annotation_file, 'w+') as fp:
        count = 0
        for abs_img_path, annotation_list in annotation_dict.items():
            fp.write('{}|'.format(abs_img_path))
            for annotation in annotation_list:
                x1 = annotation['x1']
                y1 = annotation['y1']
                x2 = annotation['x2']
                y2 = annotation['y2']
                idx = annotation['idx'] 
                fp.write(' {},{},{},{},{}'.format(x1, y1, x2, y2, idx))
            count = count + 1
            if count != len(annotation_dict):
                fp.write('\n')


def process_ijcnn():
    """
    Process on IJCNN dataset
    """
    IJCNN_ROOT = '/media/an/163EAD8F3EAD6887/DATASET/TrafficSign/IJCNN/FullIJCNN2013'

    annotation_file = os.path.join(IJCNN_ROOT, 'gt.txt')
    out_annotation_file = 'annotations/ijcnn.txt'
    annotation_dict = {}

    file_list = glob.glob(IJCNN_ROOT + '/*')

    # Read data from old annotation file
    with open(annotation_file, 'r') as fp:
        line = fp.readline()
        while line:
            data = line.strip().split(';')
            rel_img_path = data[0]
            abs_img_path = os.path.join(IJCNN_ROOT, rel_img_path)
            x1 = int(data[1])
            y1 = int(data[2])
            x2 = int(data[3])
            y2 = int(data[4])
            idx = int(data[5])

            line = fp.readline()
            # Check if the file is not exist
            if abs_img_path not in file_list:
                print("Lack file ", abs_img_path)
                continue
            
            # Save data into a dict to write to new annotation file later
            if abs_img_path not in annotation_dict:
                annotation_dict[abs_img_path] = []
            
            if idx == 33 or idx == 34:    # turn right id is 33, turn left id is 34
                annotation = {}
                annotation['x1'] = x1
                annotation['y1'] = y1
                annotation['x2'] = x2
                annotation['y2'] = y2
                annotation['idx'] = abs(idx - 34)    # turn left id 34->0, turn right id 33->1
                annotation_dict[abs_img_path].append(annotation)

    annotation_dict = {k: v for k, v in annotation_dict.items() if len(v) != 0}
    
    display_sample(annotation_dict)
    write_new_annotation(annotation_dict, out_annotation_file)

    
def process_tinghua():
    """
    Process on Tinghua dataset
    """
    TINGHUA_ROOT = '/media/an/163EAD8F3EAD6887/DATASET/TrafficSign/tinghua/tinghua/data/'

    annotation_file = os.path.join(TINGHUA_ROOT, 'annotations.json')
    out_annotation_file = 'annotations/tinghua.txt'
    annotation_dict = {}

    with open(annotation_file, 'r') as fp:
        annotations = json.load(fp)

    # Write data into a new annotation file
    with open(out_annotation_file, 'w+') as fp:
        for abs_img_path, annotation_list in annotation_dict.items():
            fp.write('{}|'.format(abs_img_path))
            for annotation in annotation_list:
                x1 = annotation['x1']
                y1 = annotation['y1']
                x2 = annotation['x2']
                y2 = annotation['y2']
                idx = annotation['idx']
                fp.write(' {},{},{},{},{}'.format(x1, y1, x2, y2, idx))
            fp.write('\n')
        imgs = annotations['imgs']

        annotation_dict = {}

        # Go through each image and search for bounding box
        for key, val in imgs.items():
            objects = val['objects']
            path = val['path']
            abs_img_path = os.path.join(TINGHUA_ROOT, path)

            annotation_list = []
            for obj in objects:
                category = obj['category']
                bbox = obj['bbox']
                annotation = {}
                idx = None
                if category == 'i10':    # 'i10' is turn right
                    idx = 1
                elif category == 'i12':  # 'i12' is turn left
                    idx = 0
                if idx is not None:
                    annotation['idx'] = idx
                    annotation['x1'] = int(bbox['xmin'])
                    annotation['y1'] = int(bbox['ymin'])
                    annotation['x2'] = int(bbox['xmax'])
                    annotation['y2'] = int(bbox['ymax'])
                    annotation_list.append(annotation)

            if len(annotation_list) != 0:
                annotation_dict[abs_img_path] = annotation_list
    
    display_sample(annotation_dict)
    write_new_annotation(annotation_dict, out_annotation_file)


def process_mtsd():
    """
    Process on MTSD dataset
    """
    MTSD_ROOT = '/media/an/163EAD8F3EAD6887/DATASET/TrafficSign/MTSD'

    split_files = glob.glob(MTSD_ROOT + '/mtsd_fully_annotated/splits/*.txt')
    image_files = glob.glob(MTSD_ROOT + '/mtsd_fully_annotated/images/*')
    annotation_files = glob.glob(MTSD_ROOT + '/mtsd_fully_annotated/annotations/*')
    out_annotation_file = 'annotations/mtsd.txt'
    annotation_dict = {}

    # print(split_files)
    # print("num image files ", len(image_files))
    # print("num annotation files ", len(annotation_files))

    ## Check all type of file in image files and annotation files
    # image_types = sets(x.split('/')[-1].split('.')[-1] for x in image_files)
    # print(image_types)
    # annotation_types  = set(x.split('/')[-1].split('.')[-1] for x in annotation_files)
    # print(annotation_types)

    
    for annotation_file in tqdm(annotation_files):
        # check if exist the correlated image file (the same name file)
        abs_img_path = annotation_file.replace('annotations', 'images').replace('.json', '.jpg')
        if abs_img_path not in image_files:
            continue

        with open(annotation_file, 'r') as fp:
            annotation = json.load(fp)
            objects = annotation['objects']

            annotation_list = []
            for obj in objects:
                bbox = obj['bbox']
                label = obj['label']
                annotation = {}
                idx = None
                if label == 'regulatory--turn-right-ahead--g1':
                    idx = 1
                elif label == 'regulatory--turn-left-ahead--g1':
                    idx = 0
                if idx is not None:
                    annotation['idx'] = idx
                    annotation['x1'] = int(bbox['xmin'])
                    annotation['y1'] = int(bbox['ymin'])
                    annotation['x2'] = int(bbox['xmax'])
                    annotation['y2'] = int(bbox['ymax'])
                    annotation_list.append(annotation)

            if len(annotation_list) != 0:
                annotation_dict[abs_img_path] = annotation_list
    
    display_sample(annotation_dict)
    write_new_annotation(annotation_dict, out_annotation_file)


def create_new_dataset(annotation_file):
    fn_annotation = annotation_file.split('/')[-1]
    name = fn_annotation.split('.')[0]

    new_img_folder = 'data/images/' + name
    new_annotation_folder = 'data/annotations/' + name

    if not os.path.exists(new_img_folder):
        os.makedirs(new_img_folder)
    if not os.path.exists(new_annotation_folder):
        os.makedirs(new_annotation_folder)

    new_annotation = []
    new_annotation_file = os.path.join(new_annotation_folder, fn_annotation)

    # Read current absoluate image path and copy to new image folder
    with open(annotation_file, 'r') as fp:
        line = fp.readline()
        while line:
            abs_img_path = line.rsplit('|', 1)[0]
            fn_img = abs_img_path.split('/')[-1]

            new_rel_img_path = os.path.join(new_img_folder, fn_img)
            shutil.copy(abs_img_path, new_rel_img_path)

            new_line = line.replace(abs_img_path, new_rel_img_path)
            new_annotation.append(new_line)
            line = fp.readline()

    # Write a new annotation file with relative image path
    with open(new_annotation_file, 'w+') as fp:
        for annotation in new_annotation:
            fp.write(annotation)


if __name__ == '__main__':
    # Create local annotation file
    # process_ijcnn()
    # process_tinghua()
    # process_mtsd()

    # Create data for online uploading
    # for annotation_file in ['annotations/ijcnn.txt', 'annotations/mtsd.txt', 'annotations/tinghua.txt']:
    #     create_new_dataset(annotation_file)
    create_new_dataset('annotations/mtsd.txt')