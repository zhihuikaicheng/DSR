from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from PIL import Image
import pdb

os.environ["CUDA_VISIBLE_DEVICES"]="0"

tfrecord_dir = '/world/data-gpu-94/sysu-reid/person-reid-data/OPPO_partial_dataset/training/'
image_dir = '/world/data-gpu-94/sysu-reid/person-reid-data/OPPO_partial_dataset_raw/training/'

with tf.Graph().as_default():
    record_files = [os.path.join(tfrecord_dir, record) for record in os.listdir(tfrecord_dir) if "tfrecords" in record]
    pdb.set_trace()
    file_queue = tf.train.string_input_producer(record_files)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    keys_to_features = {
        "image_raw": tf.FixedLenFeature((), tf.string),
        "label": tf.FixedLenFeature((), tf.int64),
        "height": tf.FixedLenFeature((), tf.int64),
        "width": tf.FixedLenFeature((), tf.int64),
        "depth": tf.FixedLenFeature((), tf.int64)
    }
    features = tf.parse_single_example(serialized_example, features=keys_to_features)
    label = features["label"]
    height = features["height"]
    width = features["width"]
    depth = features["depth"]

    def cast(feature_dict, key, dtype):
        tensor = tf.cast(feature_dict[key], dtype)
        tensor.set_shape([])
        return tensor

    height = cast(features, "height", tf.int32)
    width = cast(features, "width", tf.int32)
    depth = cast(features, "depth", tf.int32)
    label = cast(features, "label", tf.float32)

    shape = tf.stack(
        [width, height, depth]
    )

    width.set_shape([])
    height.set_shape([])
    depth.set_shape([])

    image_buffer = tf.cast(tf.image.decode_jpeg(features["image_raw"], channels=3),
                           tf.float32)
    image_buffer = tf.reshape(image_buffer, shape=shape)
    image = tf.cast(image_buffer, tf.float32)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print ("0")

        for i in range(999999):
            try:
                print ("1")
                pdb.set_trace()
                img, lab = sess.run([image, label])
                # pdb.set_trace()
                img = Image.fromarray(img, 'RGB')
                img.save(image_dir + str(lab) + '_' + str(i) + '.jpg')
            except Exception as e:
                break

        coord.request_stop()
        coord.join(threads)
