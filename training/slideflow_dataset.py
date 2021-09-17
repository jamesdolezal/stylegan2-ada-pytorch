import numpy as np
import random
from slideflow.io.reader import interleave_tfrecords
import slideflow as sf
import torch

#TODO: log the categorical outcome assignments from slideflow

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
        max_size                = None,             # Artificially limit dataset size, useful for metrics
        balance                 = None,
        annotations             = None,
        **kwargs                                    # Kwargs for Dataset base class
    ):
        self.paths = paths
        self.tile_px = tile_px
        self.rank = rank
        self.num_replicas = num_replicas
        self.augment = 'xyr' if xflip else False
        self.manifest = manifest
        self.infinite = infinite
        self.max_size = max_size
        self.balance = balance
        self.annotations = annotations
        self.seed = seed
        if self.manifest is not None:
            self.num_tiles = sum([self.manifest[t]['total'] for t in self.manifest])
        else:
            self.num_tiles = None

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
    def _parser(image, slide):
        label = [0]
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image, label

    def __iter__(self):
        dataset, _, self.num_tiles = interleave_tfrecords(self.paths,
                                                          balance=self.balance,
                                                          annotations=self.annotations,
                                                          label_parser=self._parser,
                                                          standardize=False,
                                                          augment=self.augment,
                                                          finite=(not self.infinite),
                                                          manifest=self.manifest)
        for i, (image, label) in enumerate(dataset):
            if self.max_size and i > self.max_size:
                break
            if i % self.num_replicas == self.rank:
                yield image.copy(), label
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
        max_size                = None,             # Artificially limit dataset size, useful for metrics
        **kwargs                                    # Kwargs for Dataset base class
    ):
        self.tile_px = tile_px
        self.tile_um = tile_um
        self.project_path = project_path
        self.model_type = model_type
        self.outcome_label_headers = outcome_label_headers
        self.use_labels = use_labels
        self.label_prob = None

        assert model_type in ('categorical', 'linear')

        project = sf.SlideflowProject(project_path, gpu=None)
        sf_dataset = project.get_dataset(tile_px, tile_um, filters=filters, filter_blank=filter_blank, verification=None)
        if use_labels and outcome_label_headers is not None:
            if isinstance(outcome_label_headers, list) and len(outcome_label_headers) > 1:
                raise UserError("Only one outcome_label_header is supported at a time.")
            outcome_labels, _ = sf_dataset.get_labels_from_annotations(outcome_label_headers, use_float=(model_type == 'linear'), verbose=False)
            outcome_labels = {k:v['label'] for k, v in outcome_labels.items()}
            outcome_vals = list(outcome_labels.values())
            if model_type == 'categorical':
                self.max_label = max(outcome_vals)
                self.labels = {k:to_onehot(v, self.max_label+1) for k,v in outcome_labels.items()}
                self.unique_labels = np.array(list(set(outcome_vals)))
                _all_labels = np.array(list(outcome_labels.values()))
                self.label_prob = np.array([np.sum(_all_labels == i) for i in self.unique_labels]) / len(_all_labels)
            else:
                normalized_vals = (outcome_vals - np.min(outcome_vals))/np.ptp(outcome_vals)
                self.labels = {k:[normalized_vals[i]] for i, k in enumerate(outcome_labels.keys())}
        else:
            self.max_label = 0
            self.labels = None
            outcome_labels = None

        super().__init__(
            paths=sf_dataset.get_tfrecords(),
            tile_px=tile_px,
            rank=rank,
            num_replicas=num_replicas,
            seed=seed,
            xflip=xflip,
            manifest=sf_dataset.get_manifest(),
            infinite=infinite,
            max_size=max_size,
            annotations=outcome_labels,
            balance=('BALANCE_BY_CATEGORY' if outcome_labels is not None else None),
            **kwargs
        )

    def _parser(self, image, slide):
        if self.labels is not None:
           label = self.labels[slide].copy()
        else:
            label = 0
        image = image.transpose(2, 0, 1) # HWC => CHW
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
            label = random.choices(self.unique_labels, weights=self.label_prob, k=1)[0]
            return to_onehot(label, self.max_label+1).copy()
        elif self.use_labels:
            return [np.random.rand()]
        else:
            return np.zeros((1,))