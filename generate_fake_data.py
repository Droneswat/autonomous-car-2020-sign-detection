import cv2
import numpy as np
import os
import uuid
import glob

num_gen = 100000

min_size = 10
max_size = 40

min_num = 1
max_num = 5

display = False

def perspective_transfrom(img):
    h, w, c = img.shape
    pts1 = np.float32( [[0, 0], [w, 0], [0, h], [w, h]] )
    x1 = np.random.randint(0, int(w*0.2))
    y1 = np.random.randint(0, int(h*0.2))
    x2 = np.random.randint(int(w*0.8), w+1)
    y2 = np.random.randint(0, int(h*0.2))
    x3 = np.random.randint(0, int(w*0.2))
    y3 = np.random.randint(int(h*0.8), h+1)
    x4 = np.random.randint(int(w*0.8), w+1)
    y4 = np.random.randint(int(h*0.8), h+1)
    pts2 = np.float32( [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] )

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(w, h))

    offset_x1 = min(x1, x2, x3, x4)
    offset_y1 = min(y1, y2, y3, y4)
    offset_x2 = max(x1, x2, x3, x4) - w
    offset_y2 = max(y1, y2, y3, y4) - h
    
    return dst, offset_x1, offset_y1, offset_x2, offset_y2

def contrast_light(img):
    new_img = cv2.convertScaleAbs(img, alpha=np.random.randint(80, 120)/100, beta=np.random.randint(0, 100)*-1)
    new_img[img==0] = 0
    return new_img

# dir = './fake_data'
# num_fake = 1000
# w = 512
# h = 512
# min_s = 20
# max_s = 120

# with open(dir + '/fake_annotation.txt', 'w+') as fp:
#     for i in range(0, num_fake):
#         img = np.ones((h, w, 3)).astype(np.uint8) * 255
#         num_rect = np.random.random_integers(4)
#         img_path = dir + '/' + str(i) + '.jpg'
#         fp.write(img_path)
#         fp.write('|')

#         for _ in range(1, num_rect+1):
#             x = int(np.random.random() * w)
#             y = int(np.random.random() * h)
#             s = int(np.random.random() * (max_s-min_s)) + min_s
#             x_ = min(x+s, w)
#             y_ = min(y+s, h)
#             klass = np.random.random_integers(2)

#             if klass == 1:
#                 cv2.rectangle(img, (x, y), (x_, y_), (255, 0, 0), -1)
#             else:
#                 cv2.rectangle(img, (x, y), (x_, y_), (0, 255, 0), -1)
#             fp.write(" {},{},{},{},{}".format(x, y, x_, y_,klass))
#         fp.write("\n")
        
#         cv2.imwrite(img_path, img)
#         cv2.imshow('test', img)
#         cv2.waitKey(0)

background_dir = './fake_data/background_images'
object_dir = './fake_data/objects'

out_dir = './fake_data/out'
annotation = './fake_data/annotations.txt'

left = cv2.imread(os.path.join(object_dir, 'left.jpg'))
right = cv2.imread(os.path.join(object_dir, 'right.jpg'))

count = 0 

with open(annotation, 'w+') as fp:
    while count < num_gen:
        for img_path in glob.glob(background_dir + '/*/*.jpg'):
            # generate the new out path image by uuid
            out_path = out_dir + '/' + str(uuid.uuid1()) + '.jpg'
            # save new image path to annotation file
            fp.write(out_path+'|')
            background = cv2.imread(img_path)
            h, w, c = background.shape

            fake = background.copy()
            
            # random size, x, y position for the traffic sign on background image
            num_obj = np.random.randint(min_num, max_num+1)

            for i in range(min_num, num_obj+1):
                size = np.random.randint(min_size, max_size+1)
                x1 = np.random.randint(0, w-size)
                y1 = np.random.randint(0, h-size)
                x2 = x1 + size
                y2 = y1 + size
                kls = np.random.randint(1,3)

                if kls == 1:
                    obj = cv2.resize(left, (size, size))
                else:
                    obj = cv2.resize(right, (size, size))

                # random do the perspective transform and contrast light
                obj, offset_x1, offset_y1, offset_x2, offset_y2 = perspective_transfrom(obj)
                obj = contrast_light(obj)

                # do the trick to remove the black background on object image, then overlay object on large image
                roi = fake[y1:y2, x1:x2].copy()
                roi[obj!=0] = 0
                roi = roi + obj
                fake[y1:y2, x1:x2] = roi
                
                # recalculate x1, y1, x2, y2 after do the perspective transform
                x1 = x1 + offset_x1
                y1 = y1 + offset_y1
                x2 = x2 + offset_x2
                y2 = y2 + offset_y2

                # save localization of bounding box of object with class in annotation file
                fp.write(" {},{},{},{},{}".format(x1, y1, x2, y2, kls))

            fp.write("\n")

            cv2.imwrite(out_path, fake)

            if display:
                cv2.imshow('Fake data', fake)

                k = cv2.waitKey(0)
                if k == ord('n'):
                    continue
                else:
                    count = num_gen
                    break

            count = count + 1
            if count >= num_gen:
                break

