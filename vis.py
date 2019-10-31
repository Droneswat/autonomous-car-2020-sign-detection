import cv2
import os
import time
import argparse
import glob

from lib.core.api.face_detector import FaceDetector

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def image_demo(data_dir):
    args.model
    detector = FaceDetector(args.model)

    for pic in glob.glob(data_dir + '/*'):
        if pic.endswith('jpg'):
            img = cv2.imread(pic)

            img_show = img.copy()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            boxes_class_1, boxes_class_2 = detector(img,0.3)
            print('class 1 ', boxes_class_1)
            print('class 2 ', boxes_class_2)
            if boxes_class_1.shape[0] > 0:
                for bbox in boxes_class_1:
                    cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])),
                                (int(bbox[2]), int(bbox[3])), (0, 255, 255), 4)
            if boxes_class_2.shape[0] > 0:
                for bbox in boxes_class_2:
                    cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])),
                                (int(bbox[2]), int(bbox[3])), (255, 0, 255), 4)

            cv2.namedWindow('res',0)
            cv2.imshow('res',img_show)
            k = cv2.waitKey(0)
            if k == 110:
                continue
            else:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start train.')

    parser.add_argument('--model', dest='model', type=str, default=None, help='the model to use')
    parser.add_argument('--img_dir', dest='img_dir', type=str, default=None, help='image directory')
    
    args = parser.parse_args()

    image_demo(args.img_dir)