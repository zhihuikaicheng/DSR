from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os


def batch_inputs(config, sess, num_preprocess_threads=None, is_training=True):
    """
    Contruct batches of training or evaluation examples from the image dataset.
    """
    with tf.name_scope("batch_processing"):
        filename_pattern = "train" if is_training else "val"
        record_files = [os.path.join(config["batches_dir"], record)
                        for record in os.listdir(config["batches_dir"])
                        if filename_pattern in record and "tfrecords" in record]
        dataset = tf.data.TFRecordDataset(record_files, "GZIP")
        dataset = dataset.map(
            lambda x: _parser(x, config, is_training=is_training),
            num_parallel_calls=num_preprocess_threads
        )
        dataset = dataset.shuffle(10000 + 3 * config["batch_size"])
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(config["batch_size"])
        )
        dataset = dataset.prefetch(1)
        dataset = dataset.repeat(1)
        # iterator = dataset.make_one_shot_iterator()
        iterator = dataset.make_initializable_iterator()
        sess.run(iterator.initializer)
        images, labels = iterator.get_next()
        return images, labels


def _parser(record, config, is_training=True):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature((), tf.string),
        "label": tf.FixedLenFeature((), tf.int64),
        "height": tf.FixedLenFeature((), tf.int64),
        "width": tf.FixedLenFeature((), tf.int64),
        "depth": tf.FixedLenFeature((), tf.int64)
    }

    features = tf.parse_single_example(record, features=keys_to_features)

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

    image, label = _image_preprocessing(features["image_raw"],
                                        label, width, height, depth, config,
                                        is_training=is_training)
    return image, label


def _image_preprocessing(image_raw, label, width, height, depth, config,
                         is_training=True):
    example_serialized = image_raw
    label_index = label

    label_index.set_shape([])

    shape = tf.stack(
        [width, height, depth]
    )
    width.set_shape([])
    height.set_shape([])
    depth.set_shape([])

    image_buffer = tf.cast(tf.image.decode_jpeg(example_serialized, channels=3),
                           tf.float32)
    image_buffer = tf.reshape(image_buffer, shape=shape)

    image = tf.cast(image_buffer, tf.float32)
    label = tf.cast(label, tf.int32)
    return image, label
