import tensorflow as tf
import numpy as np
import cv2
import time

from train_config import config as cfg
from lib.core.model.facebox.net import FaceBoxes

class FaceDetector:
    def __init__(self, model_path):
        """
        Arguments:
            model_path: a string, path to the model params file.
        """
        
        self.model=tf.saved_model.load(model_path)



    def __call__(self, image, score_threshold=0.5):
        """Detect faces.

        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 5].

        """

        image_fornet,scale_x,scale_y=self.preprocess(image,target_width=cfg.MODEL.win,target_height=cfg.MODEL.hin)

        image_fornet = np.expand_dims(image_fornet, 0)



        start=time.time()
        res= self.model.inference(image_fornet)

        print('xx',time.time()-start)
        boxes_class_1 = res['boxes_class_1'].numpy()
        scores_class_1 = res['scores_class_1'].numpy()
        num_boxes_class_1 = res['num_boxes_class_1'].numpy()
        print(num_boxes_class_1)

        num_boxes_class_1 = num_boxes_class_1[0]
        boxes_class_1 = boxes_class_1[0][:num_boxes_class_1]
        scores_class_1 = scores_class_1[0][:num_boxes_class_1]

        to_keep_class_1 = scores_class_1 > score_threshold
        boxes_class_1 = boxes_class_1[to_keep_class_1]
        scores_class_1 = scores_class_1[to_keep_class_1]

        boxes_class_2 = res['boxes_class_2'].numpy()
        scores_class_2 = res['scores_class_2'].numpy()
        num_boxes_class_2 = res['num_boxes_class_2'].numpy()
        print(num_boxes_class_2)

        num_boxes_class_2 = num_boxes_class_2[0]
        boxes_class_2 = boxes_class_2[0][:num_boxes_class_2]
        scores_class_2 = scores_class_2[0][:num_boxes_class_2]

        to_keep_class_2 = scores_class_2 > score_threshold
        boxes_class_2 = boxes_class_2[to_keep_class_2]
        scores_class_2 = scores_class_2[to_keep_class_2]

        ###recorver to raw image
        scaler = np.array([cfg.MODEL.hin/scale_y,
                           cfg.MODEL.win/scale_x,
                           cfg.MODEL.hin/scale_y,
                           cfg.MODEL.win/scale_x], dtype='float32')
        boxes_class_1 = boxes_class_1 * scaler
        boxes_class_2 = boxes_class_2 * scaler

        scores_class_1=np.expand_dims(scores_class_1, 0).reshape([-1,1])
        scores_class_2=np.expand_dims(scores_class_2, 0).reshape([-1,1])

        #####the tf.nms produce ymin,xmin,ymax,xmax,  swap it in to xmin,ymin,xmax,ymax
        for i in range(boxes_class_1.shape[0]):
            boxes_class_1[i] = np.array([boxes_class_1[i][1], boxes_class_1[i][0], boxes_class_1[i][3],boxes_class_1[i][2]])
        for i in range(boxes_class_2.shape[0]):
            boxes_class_2[i] = np.array([boxes_class_2[i][1], boxes_class_2[i][0], boxes_class_2[i][3],boxes_class_2[i][2]])

        return np.concatenate([boxes_class_1, scores_class_1],axis=1), np.concatenate([boxes_class_2, scores_class_2],axis=1)

    def preprocess(self,image,target_height,target_width,label=None):

        ###sometimes use in objs detects
        h,w,c=image.shape

        bimage=np.zeros(shape=[target_height,target_width,c],dtype=image.dtype)+np.array(cfg.DATA.PIXEL_MEAN,dtype=image.dtype)

        long_side=max(h,w)

        scale_x=scale_y=target_height/long_side

        image=cv2.resize(image, None,fx=scale_x,fy=scale_y)

        h_,w_,_=image.shape
        bimage[:h_, :w_, :] = image

        return bimage,scale_x,scale_y







    def init_model(self,*args):

        if len(args) == 1:
            use_pb = True
            pb_path = args[0]
        else:
            use_pb = False
            meta_path = args[0]
            restore_model_path = args[1]

        def ini_ckpt():
            graph = tf.Graph()
            graph.as_default()
            configProto = tf.ConfigProto()
            configProto.gpu_options.allow_growth = True
            sess = tf.Session(config=configProto)
            # load_model(model_path, sess)
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, restore_model_path)

            print("Model restred!")
            return (graph, sess)

        def init_pb(model_path):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            compute_graph = tf.Graph()
            compute_graph.as_default()
            sess = tf.Session(config=config)
            with tf.gfile.GFile(model_path, 'rb') as fid:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name='')

            # saver = tf.train.Saver(tf.global_variables())
            # saver.save(sess, save_path='./tmp.ckpt')
            return (compute_graph, sess)

        if use_pb:
            model = init_pb(pb_path)
        else:
            model = ini_ckpt()

        graph = model[0]
        sess = model[1]

        return graph, sess
