import cv2
import numpy as np

dir = './fake_data'
num_fake = 1000
w = 512
h = 512
min_s = 20
max_s = 120

with open(dir + '/fake_annotation.txt', 'w+') as fp:
    for i in range(0, num_fake):
        img = np.ones((h, w, 3)).astype(np.uint8) * 255
        num_rect = np.random.random_integers(4)
        img_path = dir + '/' + str(i) + '.jpg'
        fp.write(img_path)
        fp.write('|')

        for _ in range(1, num_rect+1):
            x = int(np.random.random() * w)
            y = int(np.random.random() * h)
            s = int(np.random.random() * (max_s-min_s)) + min_s
            x_ = min(x+s, w)
            y_ = min(y+s, h)
            klass = np.random.random_integers(2)

            if klass == 1:
                cv2.rectangle(img, (x, y), (x_, y_), (255, 0, 0), -1)
            else:
                cv2.rectangle(img, (x, y), (x_, y_), (0, 255, 0), -1)
            fp.write(" {},{},{},{},{}".format(x, y, x_, y_,klass))
        fp.write("\n")
        
        cv2.imwrite(img_path, img)
        # cv2.imshow('test', img)
        # cv2.waitKey(0)