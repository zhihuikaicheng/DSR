from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import decode_tfrecord as decode
import pdb
from PIL import Image
import cv2

os.environ["CUDA_VISIBLE_DEVICES"]="2"

config = {}
config['batches_dir'] = '/world/data-gpu-94/sysu-reid/person-reid-data/OPPO_partial_dataset/training/'
config['batch_size'] = 1
image_dir = '/world/data-gpu-94/sysu-reid/person-reid-data/OPPO_partial_dataset_raw/training/'

with tf.Graph().as_default():
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)

        image, label = decode.batch_inputs(config, sess)
        for i in range(9999999):
            try:
                img, lab = sess.run([image, label])
                img = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR)
                pdb.set_trace()
                cv2.imwrite(img, image_dir + str(lab) + '_' + str(i) + '.jpg')

                # pdb.set_trace()
                # img = Image.fromarray(img[0], 'RGB')
                # img.save(image_dir + str(lab) + '_' + str(i) + '.jpg')

            except Exception as e:
                break

        # coord.request_stop()
        # coord.join(threads)