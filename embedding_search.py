"""Perform an embedding search."""

import os
import pickle
import re
from functools import partial
from os.path import exists, join
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import slideflow as sf
import tensorflow as tf
import torch
from sklearn.decomposition import PCA
from slideflow.util import colors as col
from slideflow.util import log
from tqdm.auto import tqdm

import dnnlib
import legacy
import utils
from training.networks import EmbeddingGenerator, EmbeddingMappingNetwork

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    """Accepts either a comma separated list of numbers 'a,b,c' or a range 'a-c'
    and return as a list of ints.
    """

    if os.path.exists(s):
        with open(s, 'r') as f:
            return [int(i) for i in f.read().split('\n')]
    else:
        range_re = re.compile(r'^(\d+)-(\d+)$')
        m = range_re.match(s)
        if m:
            return list(range(int(m.group(1)), int(m.group(2))+1))
        vals = s.split(',')
        return [int(x) for x in vals]


def batch(iterable: Iterable, n: int = 1) -> Iterable:
    """Separates an interable into batches of maximum size `n`."""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def noise_tensor(seed: int, z_dim: int) -> torch.Tensor:
    """Creates a noise tensor based on a given seed and dimension size.

    Args:
        seed (int): Seed.
        z_dim (int): Dimension of noise vector to create.

    Returns:
        torch.Tensor: Noise vector of shape (1, z_dim)
    """
    return torch.from_numpy(np.random.RandomState(seed).randn(1, z_dim))


def load_gan_and_embeddings(
    gan_pkl: str,
    start: int,
    end: int,
    device: torch.device
) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    """Load a GAN network and create Tensor embeddings.

    Args:
        gan_pkl (str): Path to GAN network pkl.
        start (int): Starting class index.
        end (int): Ending class index.
        device (torch.device): Device.

    Returns:
        torch.nn.Module: Embedding-interpolatable GAN module.

        torch.Tensor: First class embedding.

        torch.Tensor: Second class embedding.
    """
    print('Loading networks from "%s"...' % gan_pkl)
    with dnnlib.util.open_url(gan_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    G.mapping = EmbeddingMappingNetwork(G.mapping)
    E_G = EmbeddingGenerator(G)
    label_first = torch.zeros([1, G.c_dim], device=device)
    label_first[:, start] = 1
    label_second = torch.zeros([1, G.c_dim], device=device)
    label_second[:, end] = 1
    embedding_first = G.mapping.embed(label_first).cpu().numpy()
    embedding_second = G.mapping.embed(label_second).cpu().numpy()
    embed0 = torch.from_numpy(embedding_first).to(device)
    embed1 = torch.from_numpy(embedding_second).to(device)
    return E_G, embed0, embed1

#----------------------------------------------------------------------------

class EmbeddingSearch:
    def __init__(
        self,
        E_G: torch.nn.Module,
        classifier_features: sf.model.Features,
        pca: PCA,
        device: torch.device,
        embed_first: torch.Tensor,
        embed_end: torch.Tensor,
        normalizer: Optional[sf.norm.StainNormalizer] = None,
        pca_method: str = 'delta',
        gan_kwargs: Optional[dict] = None,
    ) -> None:
        """Supervises an embedding search.

        Detailed background information is provided in the `predict()`
        function of this script.

        Args:
            E_G (torch.nn.Module): GAN generator which accepts (z, e) as input,
                where z is a noise vector and e is the class embedding vector.
            classifier_features (sf.model.Features): Function which accepts an
                image and returns a vector of features from a classifier layer.
            pca (sklearn.decomposition.PCA): Fit PCA.
            device (torch.device): PyTorch device.
            embed_first (torch.Tensor): Starting class embedding vector,
                shape=(z_dim,)
            embed_end (torch.Tensor): Ending class embedding vector,
                shape=(z_dim,)
            gan_kwargs (dict, optional): Keyword arguments for GAN.
            process_kwargs (dict, optional): Keyword arguments for
                utils.process_gan_batch().
        """
        if pca_method not in ('raw', 'delta'):
            raise ValueError("Invalid pca_method {pca_method}")
        self.E_G = E_G
        self.classifier_features = classifier_features
        self.device = device
        self.pca = pca
        self.pca_method = pca_method
        self.embed0 = embed_first
        self.embed1 = embed_end
        self.e_dim = self.embed0.shape[1]
        self.normalizer = normalizer
        self.gan_kwargs = gan_kwargs if gan_kwargs is not None else {}

    @staticmethod
    def _best_dim_by_pc_change(
        arr: np.ndarray,
        pc: int
    ) -> Tuple[int, float, float]:
        """From a list of principal component (PC) proportional changes,
        finds the index (embedding dimension) with the greatest positive change
        in the target PC while minimizing changes in other PCs.

        Args:
            arr (np.ndarray): Two-dimensional array of shape (n, num_pc),
                list of proportions that each principal component changed.
            pc (int): Index of the target principal component.

        Returns:
            int: Index of the first dimension corresponding to the greatest
            increase in the target PC with smallest change in other PCs

            float: Proportion by which the target PC changed

            float: Sum of abs(proportion) by which all other PCs changed
        """
        proportion_of_pcs = arr / np.sum(np.abs(arr), axis=-1)[:, None]
        best_prop = np.max(proportion_of_pcs[:, pc])
        idx = int(np.where(proportion_of_pcs[:, pc]==best_prop)[0])
        num_pcs = arr.shape[1]
        pc_change = arr[idx, pc]
        other_pc_change = np.sum([
            abs(arr[
                idx,
                [_p for _p in range(num_pcs) if _p != pc]
            ])
        ])
        return idx, pc_change, other_pc_change

    def plot(self) -> None:
        """Plots Principal Component (PC) changes during the embedding search.

        Args:
            pc_change (list): List of fractional changes in the target PC
                during the search as dimensions are progressively added.
            other_pc_change (list): List of the sum of fractional changes in
                all other PCs during the search as dimensions are added.
            title (str, optional): Title for plot. Defaults to None.
        """
        x = range(len(self.pc_change))
        plt.clf()
        sns.lineplot(x=x, y=self.pc_change, color='r', label='Target PC % Change')
        sns.lineplot(x=x, y=self.other_pc_change, color='b', label='Other PC % Change')
        sns.lineplot(x=x, y=self.preds, color='green', label='End prediction')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axhline(y=1, color='gray', linestyle='--')
        plt.xlabel('Search depth (dimensions)')

    def _compare_pc(
        self,
        features0: np.ndarray,
        features1: np.ndarray,
    ) -> np.ndarray:

        if self.pca_method == 'raw':
            return self.pca.transform(features1) - self.pca.transform(features0)
        elif self.pca_method == 'delta':
            return self.pca.transform(features1 - features0)

    def _features_from_embedding(
        self,
        z: torch.Tensor,
        embedding: torch.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """From a given noise vector and embedding, calculate classifier
        features and prediction.

        Args:
            z (torch.Tensor): Noise vector (from seed).
            embedding (torch.Tensor): Embedding vector.

        Returns:
            tf.Tensor: Features vector.

            tf.Tensor: Predictions vector.
        """
        if len(z.shape) == 1:
            z = torch.unsqueeze(z, axis=0)
        if len(embedding.shape) == 1:
            embedding = torch.unsqueeze(embedding, axis=0)

        start_img_batch = self.E_G(z, embedding, **self.gan_kwargs)
        start_img_batch = utils.process_gan_batch(start_img_batch)
        start_img_batch = utils.decode_batch(start_img_batch, normalizer=self.normalizer, resize_px=299)
        features0, pred0 = self.classifier_features(start_img_batch)
        pred0 = pred0[:, 0].numpy()
        features0 = features0.numpy()
        return features0, pred0

    def _create_mask(
        self,
        e: int,
        starting_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Creates an embedding mask.

        The mask is an array of ones of length embedding_dim, with the index `e`
        set to 0, multipled by a given starting mask.

        Args:
            e (int): Index of the mask to set to 0.
            starting_mask (np.ndarray, optional): Multiply the mask by this
                array. Defaults to None.

        Returns:
            np.ndarray: Embedding mask, shape=(num_embedding_dimensions,)
        """
        mask = np.ones(self.e_dim)
        mask[e] = 0
        if starting_mask is not None:
            mask *= starting_mask
        return mask

    def _embedding_from_mask(self, mask_batch: np.ndarray) -> torch.Tensor:
        mask = torch.from_numpy(mask_batch).to(self.device)
        inv_mask_batch = (~mask_batch.astype(bool)).astype(int)
        inv_mask = torch.from_numpy(inv_mask_batch).to(self.device)

        # Create images from masked embeddings.
        embed_batch = (((self.embed0.expand(len(mask_batch), -1) * mask)
                        + (self.embed1.expand(len(mask_batch), -1) * inv_mask)))
        return embed_batch

    def _full_interpolation(self, z: torch.Tensor):

        # Calculate features at the starting embedding.
        features0, _ = self._features_from_embedding(z, self.embed0)
        features1, _ = self._features_from_embedding(z, self.embed1)

        # Calculate Principal Components (PC) from a full class interpolation
        full_interp_pc = self._compare_pc(features0, features1)

        if self.pca_method == 'delta':
            # Calculate Principal Components (PC) from no interpolation
            no_interp_pc = self.pca.transform([np.zeros(self.classifier_features.num_features)])

            # Calculate difference in PCs with full interpolation
            delta_pc_full = full_interp_pc - no_interp_pc
        elif self.pca_method == 'raw':
            no_interp_pc = 0
            delta_pc_full = full_interp_pc

        return features0, no_interp_pc, delta_pc_full

    def _find_best_embedding_dim(
        self,
        z: torch.Tensor,
        pc: int,
        starting_dims: Optional[Iterable[int]] = None,
        batch_size: int = 1,
    ):
        """For a given seed `z` and target Principal Component `pc`, find the
        embedding dimension that, when traversed, maximally changes the given
        PC while minimizing changes in other PCs.

        Args:
            z (torch.Tensor): Noise vector (seed).
            pc (int): Target principal component.
            starting_dims (list(int), optional): Baseline embedding dimensions
                to traverse. If None, will not traverse any except the
                dimension being searched. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 1.

        Returns:
            int: Best embedding dimension.

            float: Amount the target PC changed.

            float: Amount all other PCs changed.
        """
        # Create starting mask
        if starting_dims is None:
            starting_dims = []
        log.debug(f"Starting with dims {starting_dims}")
        starting_mask = np.ones(self.e_dim)
        for d in starting_dims:
            starting_mask[d] = 0

        # Full interpolation, for reference.
        features0, no_interp_pc, delta_pc_full = self._full_interpolation(z)

        # Prepare masks.
        pcs = []
        preds = []
        dim_to_search = [d for d in list(range(self.e_dim)) if d not in starting_dims]
        masks = [self._create_mask(e, starting_mask=starting_mask) for e in dim_to_search]

        # GAN generator.
        def gan_generator():
            for mask_batch in batch(masks, batch_size):
                embed_batch = self._embedding_from_mask(np.stack(mask_batch))
                img_batch = self.E_G(z.expand(len(mask_batch), -1), embed_batch, **self.gan_kwargs)
                yield utils.process_gan_batch(img_batch)

        # Input data stream.
        gan_end_dts = utils.build_gan_dataset(gan_generator, 299, normalizer=self.normalizer)

        # Calculate differences while interpolating across each dimension.
        pb = tqdm(total=len(masks), leave=False, position=1, desc="Inner search")
        for img_batch in gan_end_dts:
            features_, pred_ = self.classifier_features(img_batch)
            pred_ = pred_[:, 0].numpy()
            features_ = tf.reshape(features_, (features_.shape[0], -1)).numpy()
            pc_differences = self._compare_pc(features0, features_)
            pcs += [pc_differences]
            preds += [pred_]
            pb.update(features_.shape[0])
        pb.close()

        # Calculate the effect of each embedding dimension
        # on the change in a given principal component (PC) value.
        preds = np.concatenate(preds)
        pc_by_embedding = np.concatenate(pcs)
        pc_proportion_by_embedding = (pc_by_embedding - no_interp_pc) / delta_pc_full
        dim_idx, pc_change, other_pc_change = self._best_dim_by_pc_change(pc_proportion_by_embedding, pc=pc)
        dim = dim_to_search[dim_idx]
        return dim, pc_change, other_pc_change, preds[dim_idx]

    def ordered_search(
        self,
        seed: int,
        order: Iterable[int],
        pc: int = 0,
        batch_size: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform an embedding search by progressively interpolating each
        specified embedding dimension specified in `order`.

        Use if the dimension search order is known (eg. from a previous search)

        Args:
            seed (int): Seed.
            order (list(int)): List of embedding dimensions to progressively
                transverse from class 1 -> class 2.
            pc (int, optional): _description_. Defaults to 0.
            plot (bool, optional): _description_. Defaults to False.

        Returns:
            np.ndarray: shape=(len(order), num_princpal_components).
            Proportion of target PC traversed as each dimension is added.

            np.ndarray: shape=(len(order), num_princpal_components).
            Sum of proportion of other PCs traversed as each dimension is added.

        """
        print("Performing ordered search.")
        z = noise_tensor(seed, self.E_G.z_dim)[0].to(self.device)

        # Full interpolation, for reference.
        features0, no_interp_pc, delta_pc_full = self._full_interpolation(z)

        self.pc_change = []
        self.other_pc_change = []
        self.preds = []

        # Create embedding masks.
        masks = [self._create_mask(e) for e in order]

        # GAN generator.
        def gan_generator():
            for mask_batch in batch(masks, batch_size):
                embed_batch = self._embedding_from_mask(np.stack(mask_batch))
                img_batch = self.E_G(z.expand(len(mask_batch), -1), embed_batch, **self.gan_kwargs)
                yield utils.process_gan_batch(img_batch)

        # Input data stream.
        gan_end_dts = utils.build_gan_dataset(gan_generator, 299, normalizer=self.normalizer)

        pb = tqdm(total = len(masks), leave=False)
        for img_batch in gan_end_dts:
            features_, pred_ = self.classifier_features(img_batch)
            pred_ = pred_[:, 0].numpy()
            features_ = tf.reshape(features_, (features_.shape[0], -1)).numpy()
            pc_differences = self._compare_pc(features0, features_)

            # Calculate the effect of each embedding dimension
            # on the change in a given principal component (PC) value.
            pc_proportions = (pc_differences - no_interp_pc) / delta_pc_full
            _, _pc_change, _other_pc_change = self._best_dim_by_pc_change(pc_proportions, pc=pc)
            self.pc_change += [_pc_change]
            self.other_pc_change += [_other_pc_change]
            self.preds += [pred_]
            pb.update(features_.shape[0])

        self.pc_change = np.array(self.pc_change)
        self.other_pc_change = np.array(self.other_pc_change)
        self.preds = np.concatenate(self.preds)

    def full_search(
        self,
        seed: int,
        pc: int,
        depth: Optional[int] = None,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> Tuple[List[int], List[float], List[float]]:
        """Perform a full embedding search.

        Args:
            seed (int): Seed.
            pc (int): Principal component to evaluate.
            depth (int, optional): Maximum number of dimension combinations
                to search. If None, will search through all possible dimensions
                (size of the embedding). Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 1.
            plot (bool, optional): Plot embedding search results.
                Defaults to False.

        Returns:
            List[int]: Dimensions selected, in order.

            List[float]: Proportion of target PC changed as each dimension is
            added.

            List[float]: Sum of abs(proportions) of all other target PCs changed
            as each dimension is added.
        """
        if depth is None:
            depth = self.e_dim

        print(col.bold(f"\nPerforming search for PC={pc} on seed={seed} with depth={depth}"))

        z = noise_tensor(seed, z_dim=self.E_G.z_dim)[0].to(self.device)
        self.selected_dims = []
        self.pc_change = []
        self.other_pc_change = []
        self.preds = []
        outer_pb = tqdm(total=depth, leave=False, position=0, desc="Outer search")
        for d in range(depth):
            dim, _pc_change, _other_pc_change, _preds = self._find_best_embedding_dim(
                z,
                pc=pc,
                starting_dims=self.selected_dims,
                batch_size=batch_size,
            )
            if verbose:
                tqdm.write("{}: Chose {}, {:.3f} percent PC {}, {:.3f} other PC".format(
                    col.blue(f'Depth {d}'),
                    dim,
                    _pc_change,
                    pc,
                    _other_pc_change
                ))
            self.selected_dims += [dim]
            self.pc_change += [_pc_change]
            self.other_pc_change += [_other_pc_change]
            self.preds += [_preds]
            outer_pb.update(1)
        outer_pb.close()

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--gan', 'gan_pkl', help='Network pickle filename', required=True)
@click.option('--classifier', 'classifier_path', help='Path to classifier model', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds', required=False)
@click.option('--pc-seed', type=int, help='Seed for PC search.', required=True)
@click.option('--out', type=str, help='Path to target directory in which to save plots and selections.', required=True, metavar='DIR')
@click.option('--layer', type=str, help='Layer from which to generate feature predictions.', default='postconv', show_default=True)
@click.option('--backend', type=str, default='tensorflow', help='Backend for Slideflow classifier model.')
@click.option('--verbose', type=bool, default=False, help='Verbose output.')
@click.option('--pc', type=int, required=True, help='Principal component to search.')
@click.option('--start', type=int, required=True, help='Starting class.')
@click.option('--end', type=int, required=True, help='End class.')
@click.option('--truncation-psi', type=int, default=1, required=False, help='GAN truncation PSI.')
@click.option('--noise-mode', type=str, default='const', required=False, help='GAN noise mode.')
@click.option('--resize', 'resize_method', type=str, required=False, default='tf_aa', help='Resizing method.')
@click.option('--batch-size', type=int, required=False, default=16, help='Batch size.')
@click.option('--pca-method', type=str, required=False, default='delta', help='Order in which to perform PCA, either "delta" or "raw".')
@click.option('--pca-dim', type=int, required=False, default=7, help='PCA dimensions.')
@click.option('--search', type=bool, required=False, default=True, help='Perform embedding search.')
def predict(
    ctx: click.Context,
    gan_pkl: str,
    classifier_path: str,
    seeds: num_range,
    pc_seed: int,
    out: str,
    layer: str,
    backend: str,
    verbose: bool,
    pc: int,
    start: int,
    end: int,
    truncation_psi: int,
    noise_mode: str,
    resize_method: str,
    batch_size: int,
    pca_method: str,
    pca_dim: int,
    search: bool,
):
    """Perform an embedding search.

    --- Background ------------------------------------------------------------

    The embedding search is based on a GAN -> classifier pipeline:

    \b
        GAN(seed, embedding) -> image

    where `seed` is a 512-dimension noise vector,  and `embedding` is a
    512-dimension class embedding vector.

    \b
        Classifier_Features(image) -> features

    where `image` is any image (e.g. GAN-generated image) and `features`
    is a vector of intermediate layer activations from a trained classifier.

    The merged GAN-Classifier function `f` is the following:

    \b
        f(seed, embedding) = Classifier_features(GAN(seed, embedding))

    or, expressed, another way,

    \b
        f(seed, embedding) -> features

    Using this merged GAN-Classifier function, we can calculate the change
    in the classifier feature space observed when traversing between
    class embeddings. Let `embed0` equal the class embedding from the
    starting class, and `embed1` equal the class embedding from the ending
    class:

    \b
        f(seed, embed0) -> features0
        f(seed, embed1) -> features1
        delta_features = features1 - features0

    We can then calculate `delta_features` for thousands of seeds and
    perform a PCA dimensionality reduction to find the main Principal
    Components (PC) which summarize the change in classifier feature space
    when traversing between classes while keeping the same seed.


    --- Embedding Search ------------------------------------------------------

    For a given seed, traversing between class embeddings will result in
    PC values that summarize the classifier feature space changes during
    class traversal. In an effort to explain what each PC represents,
    an "embedding search" is performed to identify which subset of embedding
    dimensions will, when traversed, change *only one of the target PCs*
    while keeping all other PCs the same. By identifying this subset of
    embedding dimensions, we can later traverse those embedding dimensions
    during interpolation and visualize how the image changes. This
    visualized change represents the changes attributable to the target PC.

    +=== NOTE! ===============================================================+
    |                                                                         |
    | This script assumes the classifier is trained to make predictions where |
    | the first class is -1 -> 0 and the second class is 0 -> 1. This is done |
    | for compatibility with the linear BRS outcome for thyroid neoplasms.    |
    | An extension of this work to support classical categorical outcomes     |
    | (discretized at 0.5 instead of 0) is forthcoming.                       |
    |                                                                         |
    +=========================================================================+

    """

    # Xception layers to try exploring:
    # --------------------------------
    # batch_normalization
    # batch_normalization_2
    # batch_normalization_3
    # block8_sepconv3_bn
    # block11_sepconv3_bn
    # postconv

    if backend not in ('torch', 'tensorflow'):
        ctx.fail("Unrecognized backend {}".format(backend))
    if backend == 'torch':
        ctx.fail("PyTorch backend not yet supported for this script.")
    if pca_method not in ("delta", "raw"):
        ctx.fail(f"Invalid pca-method '{pca_method}'")

    print(f"Performing PCA reduction on {len(seeds)} seeds.")
    print(f"Using resize method '{resize_method}'.")
    print(f"Results will be saved to {col.green(out)}")

    predictions = {0: [], 1: []}
    features = {0: [], 1: []}
    swap_labels = []
    img_seeds = []
    gan_kwargs = dict(
        truncation_psi=truncation_psi,
        noise_mode=noise_mode
    )

    # Limit GPU memory for Tensorflow and get PyTorch GPU.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    device = torch.device('cuda')

    # Load GAN network and embeddings.
    E_G, embed0, embed1 = load_gan_and_embeddings(gan_pkl, start, end, device)

    # Load model configuration.
    print('Reading model configuration...')
    config = sf.util.get_model_config(classifier_path)
    if config['hp']['normalizer']:
        normalizer = sf.norm.autoselect(
            config['hp']['normalizer'],
            config['hp']['normalizer_source']
        )
        if 'norm_fit' in config and config['norm_fit'] is not None:
            normalizer.fit(**config['norm_fit'])
    else:
        normalizer = None

    # Load classifier model and feature extractor.
    print('Loading classifier from "%s"...' % classifier_path)
    model = tf.keras.models.load_model(classifier_path)
    classifier_features = sf.model.Features.from_model(model, layers=layer, include_logits=True)

    # --- GAN-Classifier pipeline ---------------------------------------------
    def gan_generator(embedding):
        def generator():
            for seed_batch in batch(seeds, batch_size):
                z = torch.stack([noise_tensor(s, z_dim=E_G.z_dim)[0] for s in seed_batch]).to(device)
                img_batch = E_G(z, embedding.expand(z.shape[0], -1), **gan_kwargs)
                yield utils.process_gan_batch(img_batch)
        return generator

    # --- Input data stream ---------------------------------------------------
    gan_embed0_dts = utils.build_gan_dataset(gan_generator(embed0), 299, normalizer=normalizer)
    gan_embed1_dts = utils.build_gan_dataset(gan_generator(embed1), 299, normalizer=normalizer)

    # Calculate classifier features for GAN images created from seeds.
    # Calculation happens in batches to improve computational efficiency.
    # noise + embedding -> GAN -> Classifier -> Predictions, Features
    pb = tqdm(total=len(seeds), leave=False)
    for (seed_batch, embed0_batch, embed1_batch) in zip(batch(seeds, batch_size), gan_embed0_dts, gan_embed1_dts):

        features0, pred0 = classifier_features(embed0_batch)
        features1, pred1 = classifier_features(embed1_batch)
        pred0 = pred0[:, 0].numpy()
        pred1 = pred1[:, 0].numpy()
        features0 = tf.reshape(features0, (len(seed_batch), -1)).numpy().astype(np.float32)
        features1 = tf.reshape(features1, (len(seed_batch), -1)).numpy().astype(np.float32)

        # For each seed in the batch, determine if there ids "class-swapping",
        # where the GAN class label matches the classifier prediction.
        #
        # This may not happen 100% percent of the time even with a perfect GAN
        # and perfect classifier, since both classes have images that are
        # not class-specific (such as empty background, background tissue, etc)
        for i in range(len(seed_batch)):
            img_seeds += [seed_batch[i]]
            predictions[0] += [pred0[i]]
            predictions[1] += [pred1[i]]
            features[0] += [features0[i]]
            features[1] += [features1[i]]

            # NOTE: This logic assumes predictions are discretized at 0,
            # which will not be true for categorical outcomes.
            if (pred0[i] < 0) and (pred1[i] > 0):
                # Class-swapping is observed for this seed.
                if (pred0[i] < -0.5) and (pred1[i] > 0.5):
                    # Strong class swapping.
                    tail = " **"
                    swap_labels += ['strong_swap']
                else:
                    # Weak class swapping.
                    tail = " *"
                    swap_labels += ['weak_swap']
            elif (pred0[i] > 0) and (pred1[i] < 0):
                # Predictions are oppositve of what is expected.
                tail = " (!)"
                swap_labels += ['no_swap']
            else:
                tail = ""
                swap_labels += ['no_swap']
            if verbose:
                tqdm.write(f"Seed {seed_batch[i]:<6}: {pred0[i]:.2f}\t{pred1[i]:.2f}{tail}")
        pb.update(len(seed_batch))
    pb.close()

    # Convert to dataframe.
    df = pd.DataFrame({
        'seed': pd.Series(img_seeds),
        'pred_start': pd.Series(predictions[0]),
        'pred_end': pd.Series(predictions[1]),
        'features_start': pd.Series(features[0]).astype(object),
        'features_end': pd.Series(features[1]).astype(object),
        'class_swap': pd.Series(swap_labels),
    })
    swap_df = df.loc[df.class_swap.isin(['strong_swap', 'weak_swap'])]
    strong_swap_df = df.loc[df.class_swap == 'strong_swap']

    print(col.bold("\nFrequency of class-swapping: {:.2f}%".format(
        100 * (len(swap_df) / len(seeds))
    )))
    print(col.bold("Frequency of strong class-swapping: {:.2f}%\n".format(
        100 * (len(strong_swap_df) / len(seeds))
    )))

    # PCA and plots (scree plot, PC & class swap plots).
    pca = PCA(n_components=pca_dim)
    if pca_method == 'raw':
        # Fit to all features
        pca.fit(np.concatenate([features[0], features[1]]))
        all_pc_diff = pca.transform(features[1]) - pca.transform(features[0])
    elif pca_method == 'delta':
        # Fit to strong-swap features only
        ss_features_start = np.stack(strong_swap_df.features_start.to_numpy())
        ss_features_end = np.stack(strong_swap_df.features_end.to_numpy())
        pca.fit(ss_features_end - ss_features_start)
        all_pc_diff = pca.transform(np.array(features[1]) - np.array(features[0]))
    for (xpc, ypc) in [[0, n] for n in range(1, pca_dim)]:
        plt.clf()
        sns.scatterplot(
            x=all_pc_diff[:, xpc],
            y=all_pc_diff[:, ypc],
            hue=(np.array(predictions[1]) - np.array(predictions[0])),
            s=10
        )
        ax = plt.gca()
        ax.set_xlabel(f'PC{xpc}' if pca_method == 'delta' else f'Delta PC{xpc}')
        ax.set_ylabel(f'PC{ypc}' if pca_method == 'delta' else f'Delta PC{ypc}')
        plt.savefig(join(out, f'pca_scatter_{xpc}_{ypc}.svg'))

    # Calculate PC differences.
    if pca_method == 'raw':
        df.loc[:, 'pc_start'] = pd.Series(list(pca.transform(np.stack(df.features_start.to_numpy())))).astype(object)
        df.loc[:, 'pc_end'] = pd.Series(list(pca.transform(np.stack(df.features_end.to_numpy())))).astype(object)
        df.loc[:, 'pc_diff'] = df.pc_end - df.pc_start
    elif pca_method == 'delta':
        df.loc[:, 'pc_diff'] = pd.Series(list(pca.transform(
            np.stack(df.features_end.to_numpy())
            - np.stack(df.features_start.to_numpy())
        ))).astype(object)

    # Show some PC values for a handful of seeds.
    print("First 200 strong class-swap seeds:")
    for i, (_, row) in enumerate(df.loc[df.class_swap == 'strong_swap'].iterrows()):
        if i > 200:
            break
        pc_vals = ', '.join([f'{p:.3f}' for p in row.pc_diff])
        print("{}: {}".format(col.green(f"seed {row.seed}"), pc_vals))

    # Scree plot.
    pc_values = np.arange(pca.n_components_) + 1
    plt.clf()
    plt.plot(pc_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.savefig(join(out, f'{layer}_scree_{pca_method}.png'))

    if search:
        # --- Correlate embedding dimensions with Principal Components (PC) -------
        ES = EmbeddingSearch(
            E_G,
            classifier_features,
            pca=pca,
            pca_method=pca_method,
            device=device,
            embed_first=embed0,
            embed_end=embed1,
            gan_kwargs=gan_kwargs,
            normalizer=normalizer
        )
        ES.full_search(seed=pc_seed, batch_size=batch_size, pc=pc)
        # -------------------------------------------------------------------------

    # Export fit PCA.
    pca_out = join(out, 'pca.pkl')
    with open(pca_out, 'wb') as f:
        pickle.dump(pca, f)
    print(f"Wrote fit PCA to {col.green(pca_out)}")

    # Export dataframe with seeds, features and class-swap summary.
    df_out = f'{layer}_features.parquet.gzip'
    df.to_parquet(join(out, df_out), compression='gzip')
    print(f"Wrote dataframe to {col.green(df_out)}")

    # Write class-swap seeds.
    class_swap_path = join(out, 'class_swap_seeds.txt')
    with open(class_swap_path, 'w') as file:
        file.write('\n'.join([str(s) for s in swap_df.seed]))
        print(f"Wrote class-swap seeds to {col.green(class_swap_path)}")

    strong_class_swap_path = join(out, 'strong_class_swap_seeds.txt')
    with open(strong_class_swap_path, 'w') as file:
        file.write('\n'.join([str(s) for s in strong_swap_df.seed]))
        print(f"Wrote class-swap seeds to {col.green(strong_class_swap_path)}")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    predict() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
