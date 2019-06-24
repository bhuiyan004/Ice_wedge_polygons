#!/usr/bin/python3

import time
import queue
import multiprocessing
import shapefile
from skimage.measure import find_contours
import os

class Predictor(multiprocessing.Process):
    def __init__(self, input_queue, gpu_id,
                          divided_img_path,
                          POLYGON_DIR,
                          weights_path,
                          output_shp_root,
                          x_resolution,
                          y_resolution):
        multiprocessing.Process.__init__(self)
        self.input_queue = input_queue
        self.gpu_id = gpu_id

        self.divided_img_path = divided_img_path
        self.POLYGON_DIR = POLYGON_DIR
        self.weights_path = weights_path
        self.output_shp_root = output_shp_root

        self.x_resolution = x_resolution
        self.y_resolution = y_resolution

    def run(self):

        # --------------------------- Preseting --------------------------- 
        # import regular module
        import os
        import sys
        import time
        import numpy as np
        import tensorflow as tf
        import matplotlib
        import matplotlib.pyplot
        import shapefile

        # Root directory of the project
        ROOT_DIR = r"/pylon5/ps5fp1p/wez13005/local_dir"

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
	
	# import the Mask R-CNN module
        import utils
        import model as modellib
        from model import log
        # import the configuration
        import iwp_InferenceConfidenceLevel as polygon


	# --------------------------- Configurations --------------------------- 
        # Set config
        config = polygon.PolygonConfig()
        POLYGON_DIR = self.POLYGON_DIR       
        weights_path = self.weights_path
        output_shp_root = self.output_shp_root
        
        # Override the training configurations with a few
        # changes for inferencing.
        class InferenceConfig(config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        # config.display()

        # --------------------------- Preferences ---------------------------
        # Device to load the neural network on.
        # Useful if you're training a model on the same 
        # machine, in which case use CPU and leave the
        # GPU for training.
        DEVICE = "/gpu:%s"%(self.gpu_id)  # /cpu:0 or /gpu:0
        os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(self.gpu_id)

        # Inspect the model in training or inference modes
        # values: 'inference' or 'training'
        # TODO: code for 'training' test mode not ready yet
        TEST_MODE = "inference"
        
        """
        # --------------------------- Limit the GPU usable memomry --------------------------- 
        from keras.backend.tensorflow_backend import set_session
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        set_session(tf.Session(config=tf_config))
        """
        
        # --------------------------- Load validation dataset and Model ---------------------------
        # Load validation dataset
        dataset = polygon.PolygonDataset()
        dataset.load_polygon(POLYGON_DIR, "val")

        # Must call before using the dataset
        dataset.prepare()

        print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

        # Create model in inference mode
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                      config=config)
        
        # Load weights
        print("Loading weights ", weights_path)
        model.load_weights(weights_path, by_name=True)

        
        # --------------------------- Workers --------------------------- 
        while True:
            input_img  = self.input_queue.get()
            # print (input_img)
            if input_img is None:
                self.input_queue.task_done()
                print("Exiting Process %d" % self.gpu_id)
                break

            else:
                # get the upper left x y of the image
                divided_img_name = input_img.split('/')[-1]
                i,j,ul_row_divided_img,ul_col_divided_img = divided_img_name.split('.jpg')[0].split('_')
                output_shp_path = os.path.join(output_shp_root, divided_img_name.split('..jpg')[0]+'.shp')

                # read image as array
                image = matplotlib.pyplot.imread(input_img) 

                # create shp file
                w = shapefile.Writer(shapeType=shapefile.POLYGON)
                w.field('Class', 'C', size = 5)

                # ------------------ detection ---------------------
                # Run object detection
                results = model.detect([image], verbose=False)

                # Display results
                # ax = get_ax(1)
                r = results[0]

                if len(r['class_ids']):

                    # output each mask
                    for id_masks in range(r['masks'].shape[2]):

                        # read the mask
                        mask = r['masks'][:, :, id_masks]
                        padded_mask = np.zeros(
                                    (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                        padded_mask[1:-1, 1:-1] = mask

                        # the first element is the only polygon
                        contours = find_contours(padded_mask, 0.5)[0]*np.array([[self.y_resolution,self.x_resolution]])
                        # print (contours[0:20])
                        class_id = r['class_ids'][id_masks]

                        # adjust the contours to RS imagery (row,col)
                        contours = contours + np.array([[float(ul_row_divided_img),float(ul_col_divided_img)]])
                        # print (contours[0:20])
                        # swap two cols
                        contours.T[[0, 1]] = contours.T[[1, 0]]

                        # write shp file
                        w.poly(parts=[contours.tolist()])
                        w.record(class_id)

                # save shp file
                w.save(output_shp_path)


def inference_image(divided_img_path,
                    POLYGON_DIR,
                    weights_path,
                    output_shp_root,
                    x_resolution, 
                    y_resolution):

    # The number of GPU you want to use
    num_gpus = 2
    input_queue = multiprocessing.JoinableQueue()
    
    p_list = []
    
    xlist = list()
    for xfile in os.listdir(divided_img_path):
        xlist.append(os.path.join(divided_img_path, xfile))
    
    for i in range(num_gpus):
        if num_gpus == 1:
            # set the i as the GPU device you want to use
            i = 1
            p = Predictor(input_queue, i,
                          divided_img_path,
                          POLYGON_DIR,
                          weights_path,
                          output_shp_root,
                          x_resolution, 
                          y_resolution)
            p_list.append(p)
        else:
            p = Predictor(input_queue, i,
                          divided_img_path,
                          POLYGON_DIR,
                          weights_path,
                          output_shp_root,
                          x_resolution, 
                          y_resolution)
            p_list.append(p)
        
    for p in p_list:
        p.start()
    
    for job in xlist:
        input_queue.put(job)
    
    for i in range(num_gpus):
        input_queue.put(None)

    for p in p_list:
        p.join()
