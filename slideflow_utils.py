import io
import PIL
import numpy as np
import os
import tensorflow as tf
import imghdr
import json
from os.path import join, isfile
import multiprocessing as mp

FEATURE_DESCRIPTION = {'slide':    	tf.io.FixedLenFeature([], tf.string),
                       'image_raw':	tf.io.FixedLenFeature([], tf.string),
                       'loc_x':		tf.io.FixedLenFeature([], tf.int64),
                       'loc_y':		tf.io.FixedLenFeature([], tf.int64)}

FEATURE_DESCRIPTION_LEGACY =  {'slide':    tf.io.FixedLenFeature([], tf.string),
                               'image_raw':tf.io.FixedLenFeature([], tf.string)}

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

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

def tfrecord_worker(tfr_q, img_q, transformer):
    while True:
        tfrecord = tfr_q.get()
        if tfrecord is None:
            break
        else:
            dataset = tf.data.TFRecordDataset(tfrecord)
            parser = get_tfrecord_parser(tfrecord, ('image_raw',), to_numpy=True, decode_images=True)
            dataset_attrs = None
            for i, record in enumerate(dataset):
                image = parser(record)[0].numpy()
                img = transformer(image)
                if img is None:
                    continue
                 # Error check to require uniform image attributes across
                # the whole dataset.
                channels = img.shape[2] if img.ndim == 3 else 1
                cur_image_attrs = {
                    'width': img.shape[1],
                    'height': img.shape[0],
                    'channels': channels
                }
                if dataset_attrs is None:
                    dataset_attrs = cur_image_attrs
                    width = dataset_attrs['width']
                    height = dataset_attrs['height']
                    if width != height:
                        error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
                    if dataset_attrs['channels'] not in [1, 3]:
                        error('Input images must be stored as RGB or grayscale')
                    if width != 2 ** int(np.floor(np.log2(width))):
                        error('Image width/height after scale and crop are required to be power-of-two')
                elif dataset_attrs != cur_image_attrs:
                    err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
                    error(f'Image {tfrecord} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

                # Save the image as an uncompressed PNG.
                img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB' }[channels])
                image_bits = io.BytesIO()
                img.save(image_bits, format='png', compress_level=0, optimize=False)

                img_q.put({'buffer': image_bits, 'label': None})
            img_q.put(None)

def slideflow_iterator(directory, transform_img):
    import tensorflow as tf
    num_cores = 16
    tfrecords = [f for f in os.listdir(directory) if isfile(join(directory, f)) and path_to_ext(f) == 'tfrecords']
    tfrecord_paths = [join(directory, tfr) for tfr in tfrecords]
    with open(join(directory, 'manifest.json'), 'r') as manifest_file:
        manifest = json.load(manifest_file)

    num_tiles = sum([manifest[tfr]['total'] for tfr in tfrecords])

    def generator():
        tfr_q = mp.Queue()
        img_q = mp.Queue()
        pool = mp.Pool(num_cores, initializer=tfrecord_worker, initargs=(tfr_q, img_q, transform_img))
        tfrecords_finished = 0
        for tfrecord in tfrecord_paths:
            tfr_q.put(tfrecord)
        for _ in range(num_cores):
            tfr_q.put(None)
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
        pool.close()
        pool.join()

    return num_tiles, generator()