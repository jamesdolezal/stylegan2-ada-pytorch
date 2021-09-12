import queue
import os

'''logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from slideflow.io.tfrecords import update_manifest_at_dir
del os.environ['CUDA_VISIBLE_DEVICES']'''

import imghdr
import json
from os.path import join, isfile, isdir, exists
import threading


FEATURE_DESCRIPTION = {'slide':    	tf.io.FixedLenFeature([], tf.string),
                       'image_raw':	tf.io.FixedLenFeature([], tf.string),
                       'loc_x':		tf.io.FixedLenFeature([], tf.int64),
                       'loc_y':		tf.io.FixedLenFeature([], tf.int64)}

FEATURE_DESCRIPTION_LEGACY =  {'slide':    tf.io.FixedLenFeature([], tf.string),
                               'image_raw':tf.io.FixedLenFeature([], tf.string)}

# ---------------------------------------------------------------------------

def detect_tfrecord_format(tfr):
    record = next(iter(tf.data.TFRecordDataset(tfr)))
    try:
        features = tf.io.parse_single_example(record, FEATURE_DESCRIPTION)
        for feature in FEATURE_DESCRIPTION:
            if feature not in features:
                raise tf.errors.InvalidArgumentError
        feature_description = FEATURE_DESCRIPTION
    except tf.errors.InvalidArgumentError:
        try:
            features = tf.io.parse_single_example(record, FEATURE_DESCRIPTION_LEGACY)
            feature_description = FEATURE_DESCRIPTION_LEGACY
        except tf.errors.InvalidArgumentError:
            raise Xception(f"Unrecognized TFRecord format: {tfr}")
    image_type = imghdr.what('', features['image_raw'].numpy())
    return feature_description, image_type

# ---------------------------------------------------------------------------

def _decode_image(img_string,
                  img_type,
                  size=None,
                  standardize=False,
                  normalizer=None,
                  augment=False):

    tf_decoders = {
        'png': tf.image.decode_png,
        'jpeg': tf.image.decode_jpeg,
        'jpg': tf.image.decode_jpeg
    }
    decoder = tf_decoders[img_type.lower()]
    image = decoder(img_string, channels=3)

    if normalizer:
        image = tf.py_function(normalizer.tf_to_rgb, [image], tf.int32)
        if size: image.set_shape([size, size, 3])
    if augment:
        # Augment with random compession
        image = tf.cond(tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32) < 0.5,
                        true_fn=lambda: tf.image.adjust_jpeg_quality(image, tf.random.uniform(shape=[],
                                                                                              minval=50,
                                                                                              maxval=100,
                                                                                              dtype=tf.int32)),
                        false_fn=lambda: image)

        # Rotate randomly 0, 90, 180, 270 degrees
        image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        # Random flip and rotation
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    if standardize:
        image = tf.image.per_image_standardization(image)
    if size:
        image.set_shape([size, size, 3])
    return image

# ---------------------------------------------------------------------------

def get_tfrecord_parser(tfrecord_path,
                        features_to_return=None,
                        to_numpy=False,
                        decode_images=True,
                        standardize=False,
                        img_size=None,
                        normalizer=None,
                        augment=False,
                        error_if_invalid=True):

    feature_description, img_type = detect_tfrecord_format(tfrecord_path)
    if features_to_return is None:
        features_to_return = list(feature_description.keys())

    def parser(record):
        features = tf.io.parse_single_example(record, feature_description)

        def process_feature(f):
            if f not in features and error_if_invalid:
                raise Exception(f"Unknown feature {f}")
            elif f not in features:
                return None
            elif f == 'image_raw' and decode_images:
                return _decode_image(features['image_raw'], img_type, img_size, standardize, normalizer, augment)
            elif to_numpy:
                return features[f].numpy()
            else:
                return features[f]

        if type(features_to_return) == dict:
            return {label: process_feature(f) for label, f in features_to_return.items()}
        else:
            return [process_feature(f) for f in features_to_return]

    return parser

#----------------------------------------------------------------------------

def path_to_ext(path):
    '''Returns extension of a file path string.'''
    _file = path.split('/')[-1]
    if len(_file.split('.')) == 1:
        return ''
    else:
        return _file.split('.')[-1]

#----------------------------------------------------------------------------

def slideflow_iterator(directory, transform_img=None, subfolder_labels=False):
    num_threads = 16
    num_tiles = 0
    labels = {}
    label_idx = {}
    tfrecords, tfrecord_paths = [], []
    if subfolder_labels:
        subfolders = [d for d in os.listdir(directory) if isdir(join(directory, d))]
        for i, name in enumerate(subfolders):
            subdir = join(directory, name)
            subfolder_tfr = [f for f in os.listdir(subdir) if isfile(join(subdir, f)) and path_to_ext(f) == 'tfrecords']
            subfolder_tfr_paths = [join(subdir, tfr) for tfr in subfolder_tfr]
            tfrecords += subfolder_tfr
            tfrecord_paths += subfolder_tfr_paths
            if not exists(join(subdir, 'manifest.json')):
                update_manifest_at_dir(subdir)
            with open(join(subdir, 'manifest.json'), 'r') as manifest_file:
                manifest = json.load(manifest_file)
            num_tiles += sum([manifest[tfr]['total'] for tfr in subfolder_tfr])
            labels.update({tfr: i for tfr in subfolder_tfr_paths})
            label_idx[name] = i
        print("Assigned labels:", label_idx)
    else:
        tfrecords = [f for f in os.listdir(directory) if isfile(join(directory, f)) and path_to_ext(f) == 'tfrecords']
        tfrecord_paths = [join(directory, tfr) for tfr in tfrecords]
        with open(join(directory, 'manifest.json'), 'r') as manifest_file:
            manifest = json.load(manifest_file)
    print(labels)

    def generator():

        task_finished = False

        def tfrecord_worker(tfr_q, img_q, transformer=None):
            while True:
                try:
                    tfrecord = tfr_q.get()
                    dataset = tf.data.TFRecordDataset(tfrecord)
                    parser = get_tfrecord_parser(tfrecord, ('image_raw',), to_numpy=True, decode_images=False)
                    for record in dataset:
                        img = parser(record)[0]
                        if transformer is not None:
                            img = transformer(img)
                        if img is None:
                            continue
                        image_bits = img
                        label = None if tfrecord not in labels else labels[tfrecord]
                        img_q.put({'buffer': image_bits, 'label': label})
                    img_q.put(None)
                    tfr_q.task_done()
                except queue.Empty:
                    if task_finished:
                        return
                except:
                    img_q.put(None)
                    tfr_q.task_done()

        tfr_q = queue.Queue()
        img_q = queue.Queue(1024)
        threads = [threading.Thread(target=tfrecord_worker, daemon=True, args=(tfr_q, img_q)) for t in range(num_threads)]
        for thread in threads:
            thread.start()

        tfrecords_finished = 0
        for tfrecord in tfrecord_paths:
            tfr_q.put(tfrecord)

        while True:
            image = img_q.get()
            if image is None:
                tfrecords_finished += 1
                if tfrecords_finished == len(tfrecords):
                    break
                else:
                    continue
            else:
                yield image
        task_finished = True

    return num_tiles, generator()