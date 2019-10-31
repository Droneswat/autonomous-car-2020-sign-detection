import cv2
import numpy as np

annotation_file = "train.txt"

with open(annotation_file, "r") as fp:
    line = fp.readline()
    while line:
        line=line.rstrip()

        img_path=line.rsplit('| ',1)[0]
        content=line.rsplit('| ',1)[-1]
        labels = content.split(' ')
        image = cv2.imread(img_path)
        
        for label in labels:
            bbox = np.array(label.split(','), dtype=np.float)
            print(bbox)
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 255), 10)

        image = cv2.resize(image, (256, 256))
        cv2.imshow("Check", image)
        
        line = fp.readline()
        k = cv2.waitKey(0)
        if k == 110:
            continue
        else:
            break