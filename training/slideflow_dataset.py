import numpy as np
import os
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
assert(not tf.test.is_gpu_available())
import slideflow.io.tfrecords
import slideflow as sf
import torch
del os.environ['CUDA_VISIBLE_DEVICES']

def to_onehot(val, max):
    onehot = np.zeros(max, dtype=np.int64)
    onehot[val] = 1
    return onehot

class UserError(Exception):
    pass

class InterleaveIterator(torch.utils.data.IterableDataset):
    def __init__(self,
        paths,                                      # Path to tfrecord files to interleave
        tile_px,                                    # Image width in pixels
        rank                    = 0,                # Which GPU replica this dataset is being used for
        num_replicas            = 1,                # Total number of GPU replicas
        seed                    = None,             # Tensorflow seed for random sampling
        xflip                   = False,            # Bool indicating whether data should be augmented (flip/rotate)
        manifest                = None,             # Manifest mapping tfrecord names to number of total tiles
        infinite                = True,             # Inifitely loop through dataset
        **kwargs                                    # Kwargs for Dataset base class
    ):
        self.paths = paths
        self.tile_px = tile_px
        self.rank = rank
        self.num_replicas = num_replicas
        self.augment = 'xyr' if xflip else False
        self.manifest = manifest
        self.infinite = infinite
        if self.manifest is not None:
            self.num_tiles = sum([self.manifest[t]['total'] for t in self.manifest])
        else:
            self.num_tiles = None


        if seed is not None:
            tf.random.set_seed(seed)

    @property
    def name(self):
        return 'slideflow-test'#self._name

    @property
    def resolution(self):
        return self.tile_px

    @property
    def image_shape(self):
        return (3, self.resolution, self.resolution)

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def label_shape(self):
        return 0

    @property
    def label_dim(self):
        return 0

    @property
    def has_labels(self):
        return False

    @property
    def has_onehot_labels(self):
        return False

    @staticmethod
    def _parser(record, base_parser, **kwargs):
        slide, image = base_parser(record)
        label = [0]
        image = tf.transpose(image, perm=(2, 0, 1)) # HWC => CHW
        return image, label

    def __iter__(self):
        dataset, _, self.num_tiles = sf.io.tfrecords.interleave_tfrecords(self.paths,
                                                                          image_size=self.resolution,
                                                                          batch_size=None,
                                                                          label_parser=self._parser,
                                                                          standardize=False,
                                                                          augment=self.augment,
                                                                          finite=(not self.infinite),
                                                                          manifest=self.manifest)

        for i, (image, label) in enumerate(dataset):
            if i % self.num_replicas == self.rank:
                yield image.numpy(), label.numpy()
            else:
                continue

    def close(self):
        pass

    def get_details(self, idx):
        raise NotImplementedError

    def get_label(self, idx):
        raise NotImplementedError

class SlideflowIterator(InterleaveIterator):
    def __init__(self,
        tile_px,                                    # Image width in pixels
        tile_um,                                    # Image width in microns
        project_path,                               # Path to slideflow project directory
        model_type              = 'categorical',    # Indicates type of outcome label, 'categorical' or 'linear'
        outcome_label_headers   = None,             # Annotations column header indicating outcome
        filters                 = None,             # Slideflow dataset arg `filters`
        filter_blank            = None,             # Slideflow dataset arg `filter_blank`
        rank                    = 0,                # Which GPU replica this dataset is being used for
        num_replicas            = 1,                # Total number of GPU replicas
        seed                    = None,             # Tensorflow seed for random sampling
        xflip                   = False,            # Bool indicating whether data should be augmented (flip/rotate)
        use_labels              = False,            # Enable conditioning labels?
        infinite                = True,             # Infinite dataset looping
        **kwargs                                    # Kwargs for Dataset base class
    ):
        self.tile_px = tile_px
        self.tile_um = tile_um
        self.project_path = project_path
        self.model_type = model_type
        self.outcome_label_headers = outcome_label_headers
        self.use_labels = use_labels

        assert model_type in ('categorical', 'linear')

        project = sf.SlideflowProject(project_path, gpu=None)
        sf_dataset = project.get_dataset(tile_px, tile_um, filters=filters, filter_blank=filter_blank, verification=None)
        if use_labels and outcome_label_headers is not None:
            if isinstance(outcome_label_headers, list) and len(outcome_label_headers) > 1:
                raise UserError("Only one outcome_label_header is supported at a time.")
            outcome_labels, _ = sf_dataset.get_labels_from_annotations(outcome_label_headers, use_float=(model_type == 'linear'))
            outcome_labels = {k:v['label'] for k, v in outcome_labels.items()}
            outcome_vals = np.array(list(outcome_labels.values()))
            if model_type == 'categorical':
                self.max_label = np.max(outcome_vals)
                self.labels = {k:to_onehot(v, self.max_label+1) for k,v in outcome_labels.items()}
                self._all_labels = list(self.labels.values())
            else:
                normalized_vals = (outcome_vals - np.min(outcome_vals))/np.ptp(outcome_vals)
                self.labels = {k:[normalized_vals[i]] for i, k in enumerate(outcome_labels.keys())}

        else:
            self.max_label = 0
            self.labels = None

        super().__init__(
            paths=sf_dataset.get_tfrecords(),
            tile_px=tile_px,
            rank=rank,
            num_replicas=num_replicas,
            seed=seed,
            xflip=xflip,
            manifest=sf_dataset.get_manifest(),
            infinite=infinite,
            **kwargs
        )

    def _parser(self, record, base_parser, **kwargs):
        slide, image = base_parser(record)
        if self.labels is not None:
            def label_lookup(s): return self.labels[s.numpy().decode('utf-8')]
            label_dtype = tf.int64 if self.model_type == 'categorical' else tf.float32
            label = tf.py_function(func=label_lookup,
                                    inp=[slide],
                                    Tout=label_dtype)
        else:
            label = 0
        image = tf.transpose(image, perm=(2, 0, 1)) # HWC => CHW
        return image, label

    @property
    def label_shape(self):
        if self.use_labels and self.model_type == 'categorical':
            return (self.max_label+1,)
        elif self.use_labels:
            return (1,)
        else:
            return 0

    @property
    def label_dim(self):
        if self.use_labels:
            assert len(self.label_shape) == 1
            return self.label_shape[0]
        else:
            return 0

    @property
    def has_labels(self):
        return self.use_labels and any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

    def get_label(self, idx):
        if self.use_labels and self.model_type == 'categorical':
            return self._all_labels[np.random.choice(range(len(self._all_labels)))]
        elif self.use_labels:
            return [np.random.rand()]
        else:
            return np.zeros((1,))