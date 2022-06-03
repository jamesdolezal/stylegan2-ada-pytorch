"""Perform an embedding search."""

import os
import re
from os.path import exists, join
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

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
from tqdm import tqdm

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
        gan_kwargs: Optional[dict] = None,
        process_kwargs: Optional[dict] = None,
        pca_method: str = 'delta',
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
        self.device = device
        self.classifier_features = classifier_features
        self.pca = pca
        self.pca_method = pca_method
        self.embed0 = embed_first
        self.embed1 = embed_end
        self.e_dim = self.embed0.shape[1]
        self.gan_kwargs = gan_kwargs if gan_kwargs is not None else {}
        self.process_kwargs = process_kwargs if process_kwargs is not None else {}

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

    @staticmethod
    def _plot_pc_change(
        pc_change: Iterable[float],
        other_pc_change: Iterable[float],
        title: Optional[str] = None
    ) -> None:
        """Plots Principal Component (PC) changes during the embedding search.

        Args:
            pc_change (list): List of fractional changes in the target PC
                during the search as dimensions are progressively added.
            other_pc_change (list): List of the sum of fractional changes in
                all other PCs during the search as dimensions are added.
            title (str, optional): Title for plot. Defaults to None.
        """
        x = range(len(pc_change))
        if title is None:
            title = 'Embedding Search Plot'
        plt.clf()
        plt.title(title)
        sns.lineplot(x=x, y=pc_change, color='r')
        sns.lineplot(x=x, y=other_pc_change, color='b')
        plt.axhline(y=1, color='black', linestyle='--')
        plt.show()

    def _compare_pc(
        self,
        z: torch.Tensor,
        embedding_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """For a given seed `z`, compare the dimensionality-reduced principal
        components (PC) for the classifier feature space between classes,
        where classifier features are generated for a given seed by passing
        a GAN-generated image for a seed and class embedding (multiplied by a
        mask) to the model classifier.

        Args:
            z (torch.Tensor): Seed. Either single or batch.
            embedding_mask (np.ndarray, optional): Mask(s) to be applied to the
                first embedding. Either single or batch. Inverse of the mask is
                applied to the second embedding. Defaults to None
                (full interpolation).

        Returns:
            np.ndarray: Differences in PC between classifiers,
            with shape=(num_embedding_dimensions, num_principal_components)
        """
        if embedding_mask is None:
            embedding_mask = np.zeros(self.e_dim)
        if not len(z.shape) == len(embedding_mask.shape):
            raise ValueError('z and embedding_shape must have the same number '
                             f'of dimensions (got {len(z.shape)} and '
                             f'{len(embedding_mask.shape)})')
        if len(z.shape) == 1:
            z = torch.unsqueeze(z, axis=0)
            embedding_mask = np.expand_dims(embedding_mask, axis=0)

        batch_size = z.shape[0]

        # Create embedding masks.
        inv_embedding_mask = (~embedding_mask.astype(bool)).astype(int)
        mask = torch.from_numpy(embedding_mask).to(self.device)
        inv_mask = torch.from_numpy(inv_embedding_mask).to(self.device)

        # Create images from masked embeddings.
        start_embed = self.embed0.expand(batch_size, -1)
        end_embed = (((self.embed0.expand(batch_size, -1) * mask)
                    + (self.embed1.expand(batch_size, -1) * inv_mask)))
        img0 = self.E_G(z, start_embed, **self.gan_kwargs)
        img1 = self.E_G(z, end_embed, **self.gan_kwargs)

        # Process a batch of GAN images, applying crop/resizing operations
        # and image normalization.
        image0 = utils.process_gan_batch(img0, **self.process_kwargs)
        image1 = utils.process_gan_batch(img1, **self.process_kwargs)

        # Calculate features from generated images.
        features0 = self.classifier_features(image0).numpy().reshape(batch_size, -1)
        features1 = self.classifier_features(image1).numpy().reshape(batch_size, -1)

        # Reduce features to principal components.
        if self.pca_method == 'raw':
            pc1 = self.pca.transform(features1)
            pc0 = self.pca.transform(features0)
            pc = pc1 - pc0
        elif self.pca_method == 'delta':
            pc = self.pca.transform(features1 - features0)

        return pc

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

    def _find_best_embedding_dim(
        self,
        z: torch.Tensor,
        pc: int,
        starting_dims: Optional[Iterable[int]] = None,
        batch_size: int = 1
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

        # Calculate Principal Components (PC) from a full class interpolation
        full_interp_pc = self._compare_pc(z)

        if self.pca_method == 'delta':
            # Calculate Principal Components (PC) from no interpolation
            no_interp_pc = self.pca.transform([np.zeros(self.classifier_features.num_features)])

            # Calculate difference in PCs with full interpolation
            delta_pc_full = full_interp_pc - no_interp_pc
        elif self.pca_method == 'raw':
            no_interp_pc = np.zeros(full_interp_pc.shape[0])
            delta_pc_full = full_interp_pc

        pcs = []
        dim_to_search = [
            d for d in list(range(self.e_dim))
            if d not in starting_dims
        ]
        masks = [
            self._create_mask(embed_idx, starting_mask=starting_mask)
            for embed_idx in dim_to_search
        ]
        pb = tqdm(total=len(masks), leave=False, position=0)
        for mask_batch in batch(masks, batch_size):
            pc_differences = self._compare_pc(
                z=z.expand(len(mask_batch), -1),
                embedding_mask=np.array(mask_batch)
            )
            pcs += [pc_differences]
            pb.update(len(mask_batch))
        pb.close()

        # Calculate the effect of each embedding dimension
        # on the change in a given principal component (PC) value.
        pc_by_embedding = np.concatenate(pcs)
        pc_proportion_by_embedding = (pc_by_embedding - no_interp_pc) / delta_pc_full
        dim_idx, pc_change, other_pc_change = self._best_dim_by_pc_change(pc_proportion_by_embedding, pc=pc)
        dim = dim_to_search[dim_idx]
        return dim, pc_change, other_pc_change

    def ordered_search(
        self,
        seed: int,
        order: Iterable[int],
        pc: int = 0,
        plot: bool = False
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
        mask = np.ones(self.e_dim)

        # Calculate Principal Components (PC) from a full class interpolation
        full_interp_pc = self._compare_pc(z)

        # Calculate Principal Components (PC) from no interpolation
        no_interp_pc = self.pca.transform([np.zeros(self.classifier_features.num_features)])

        # Calculate difference in PCs with full interpolation
        delta_pc_full = full_interp_pc - no_interp_pc

        all_pc_change = []
        all_other_pc_change = []
        for e in tqdm(order, leave=False):
            mask[e] = 0
            pc_differences = self._compare_pc(z, embedding_mask=mask)

            # Calculate the effect of each embedding dimension
            # on the change in a given principal component (PC) value.
            pc_proportions = (pc_differences - no_interp_pc) / delta_pc_full
            _, pc_change, other_pc_change = self._best_dim_by_pc_change(pc_proportions, pc=pc)
            all_pc_change += [pc_change]
            all_other_pc_change += [other_pc_change]

        all_pc_change = np.array(all_pc_change)
        all_other_pc_change = np.array(all_other_pc_change)
        if plot:
            self._plot_pc_change(
                all_pc_change,
                all_other_pc_change,
                title=f'Embedding Search Plot (PC={pc}, seed={seed})'
            )
        return all_pc_change, all_other_pc_change

    def full_search(
        self,
        seed: int,
        pc: int,
        depth: Optional[int] = None,
        batch_size: int = 1,
        plot: bool = False
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
        selected_dims = []
        all_pc_change = []
        all_other_pc_change = []
        outer_pb = tqdm(total=depth, leave=False, position=1)
        for d in range(depth):
            dim, pc_change, other_pc_change = self._find_best_embedding_dim(
                z,
                pc=pc,
                starting_dims=selected_dims,
                batch_size=batch_size
            )
            print("{}: Chose {}, {:.3f} percent PC {}, {:.3f} other PC".format(
                col.blue(f'Depth {d}'),
                dim,
                pc_change,
                pc,
                other_pc_change
            ))
            selected_dims += [dim]
            all_pc_change += [pc_change]
            all_other_pc_change += [other_pc_change]
            outer_pb.update(1)
        outer_pb.close()
        print("Order of embedding dimension selection:", selected_dims)
        if plot:
            self._plot_pc_change(
                all_pc_change,
                all_other_pc_change,
                title=f'Embedding Search Plot (PC={pc}, seed={seed})'
            )
        return selected_dims, all_pc_change, all_other_pc_change


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
@click.option('--lasso', type=bool, default=False, help='Interactive lasso plot.')
@click.option('--verbose', type=bool, default=False, help='Verbose output.')
@click.option('--pc', type=int, required=True, help='Principal component to search.')
@click.option('--start', type=int, required=True, help='Starting class.')
@click.option('--end', type=int, required=True, help='End class.')
@click.option('--truncation-psi', type=int, default=1, required=False, help='GAN truncation PSI.')
@click.option('--noise-mode', type=str, default='const', required=False, help='GAN noise mode.')
@click.option('--resize', 'resize_method', type=str, required=False, default='tf_aa', help='Resizing method.')
@click.option('--batch-size', type=int, required=False, default=16, help='Batch size.')
@click.option('--export', type=bool, required=False, default=False, help='Export results.')
@click.option('--scree', type=bool, required=False, default=False, help='Export results.')
@click.option('--pca-method', type=str, required=False, default='delta', help='Order in which to perform PCA, either "delta" or "raw".')
@click.option('--pca-dim', type=int, required=False, default=7, help='PCA dimensions.')
def predict(
    ctx: click.Context,
    gan_pkl: str,
    classifier_path: str,
    seeds: num_range,
    pc_seed: int,
    out: str,
    layer: str,
    backend: str,
    lasso: bool,
    verbose: bool,
    pc: int,
    start: int,
    end: int,
    truncation_psi: int,
    noise_mode: str,
    resize_method: str,
    batch_size: int,
    export: bool,
    scree: bool,
    pca_method: str,
    pca_dim: int,
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

    print(f"Detected {len(seeds)} seeds")
    print(f"Using resize method '{resize_method}'")

    class_swap_seeds = []
    class_swap_features = []
    strong_class_swap_seeds = []
    calculated_features = []
    plot_labels = []
    pred_labels = []
    swap_labels = []
    gan_labels = []
    img_seeds = []
    used_embed = []
    gan_kwargs = dict(
        truncation_psi=truncation_psi,
        noise_mode=noise_mode
    )

    # Limit GPU memory for Tensorflow.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load GAN network and embeddings.
    print('Loading networks from "%s"...' % gan_pkl)
    device = torch.device('cuda')
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
    process_kwargs = dict(
        normalizer=normalizer,
        resize_method=resize_method
    )

    # Load classifier model and feature extractor.
    print('Loading classifier from "%s"...' % classifier_path)
    model = tf.keras.models.load_model(classifier_path)
    classifier_features = sf.model.Features.from_model(model, layers=layer)

    # Calculate classifier features for GAN images created from seeds.
    # Calculation happens in batches to improve computational efficiency.
    pb = tqdm(total=len(seeds), leave=False)
    for seed_batch in batch(seeds, batch_size):
        n_seeds = len(seed_batch)
        for s in seed_batch:
            img_seeds += [s, s]

        # Create and process GAN images.
        z = torch.stack([noise_tensor(s, z_dim=E_G.z_dim)[0] for s in seed_batch]).to(device)
        img0_batch = E_G(z, embed0.expand(n_seeds, -1), **gan_kwargs)
        img1_batch = E_G(z, embed1.expand(n_seeds, -1), **gan_kwargs)
        img0_batch = utils.process_gan_batch(img0_batch, **process_kwargs)
        img1_batch = utils.process_gan_batch(img1_batch, **process_kwargs)

        # Create model predictions from GAN images.
        pred0 = model.predict(img0_batch)[:, 0]
        pred1 = model.predict(img1_batch)[:, 0]
        features0 = tf.reshape(classifier_features(img0_batch), (n_seeds, -1)).numpy()
        features1 = tf.reshape(classifier_features(img1_batch), (n_seeds, -1)).numpy()

        # For each seed in the batch, determine if there is "class-swapping",
        # where the GAN class label matches the classifier prediction.
        #
        # This may not happen 100% percent of the time even with a perfect GAN
        # and perfect classifier, since both classes have images that are
        # not class-specific (such as empty background, background tissue, etc)
        for i in range(n_seeds):
            pred_labels += [pred0[i], pred1[i]]
            calculated_features += [features0[i], features1[i]]
            used_embed += [0, 1]
            plot_labels += [
                ('Braf-like' if pred0[i] < 0 else 'Ras-like'),
                ('Braf-like' if pred1[i] < 0 else 'Ras-like')
            ]
            gan_labels += ['GAN-Braf-like', 'GAN-Ras-like']
            # NOTE: This logic assumes predictions are discretized at 0,
            # which will not be true for categorical outcomes.
            if (pred0[i] < 0) and (pred1[i] > 0):
                # Class-swapping is observed for this seed.
                class_swap_seeds += [seed_batch[i]]
                class_swap_features += [features0[i], features1[i]]
                class_swap_feature_differences += [features1[i] - features0[i]]

                if (pred0[i] < -0.5) and (pred1[i] > 0.5):
                    # Strong class swapping.
                    tail = "**"
                    strong_class_swap_seeds += [seed_batch[i]]

                    # Calculate features for each class
                    strong_class_swap_features0 += [features0[i]]
                    strong_class_swap_features1 += [features1[i]]
                    swap_labels += ['strong_swap', 'strong_swap']
                else:
                    # Weak class swapping.
                    tail = "*"
                    swap_labels += ['weak_swap', 'weak_swap']
            elif (pred0[i] > 0) and (pred1[i] < 0):
                # Predictions are oppositve of what is expected.
                tail = " (!)"
                swap_labels += ['no_swap', 'no_swap']
            else:
                tail = ""
                swap_labels += ['no_swap', 'no_swap']
            if verbose:
                tqdm.write(f"Seed {seed_batch[i]:<6}: {pred0[i]:.2f}\t{pred1[i]:.2f}{tail}")
        pb.update(n_seeds)
    pb.close()
    print(col.bold("\nFrequency of class-swapping: {:.2f}%\n".format(100 * (len(class_swap_seeds) / len(seeds)))))
    print(col.bold("\nFrequency of strong class-swapping: {:.2f}%\n".format(100 * (len(strong_class_swap_seeds) / len(seeds)))))

    # PCA and scree plot.
    pca = PCA(n_components=pca_dim)
    if pca_method == 'raw':
        pca.fit(calculated_features)
        features_to_reduce = map(lambda x: x[1] - x[0], batch(strong_class_swap_seeds, 2))
    elif pca_method == 'delta':
        pca.fit(class_swap_feature_differences)
        features_to_reduce = class_swap_feature_differences

    # Show some PC values for a handful of seeds.
    print("First 200 strong class-swap seeds:")
    for i, (seed, feat_diff) in enumerate(zip(strong_class_swap_seeds, features_to_reduce)):
        pca_transformed = pca.transform([feat_diff])
        pc_vals = ', '.join([f'{p:.3f}' for p in pca_transformed[0]])
        print("{}: {}".format(col.green(f"seed {seed}"), pc_vals))
        if i > 200:
            break

    pc_values = np.arange(pca.n_components_) + 1
    if scree:
        plt.plot(pc_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
        plt.title('Scree Plot')
        plt.show()

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
        process_kwargs=process_kwargs
    )
    ES.full_search(seed=pc_seed, batch_size=batch_size, pc=pc, plot=True)
    # -------------------------------------------------------------------------

    # Generate UMAP.
    umap = sf.stats.gen_umap(calculated_features, densmap=True, n_neighbors=30)

    # Export dataframe with seeds, features and class-swap summary.
    if export:
        calc_feat = [cf.astype(np.float32) for cf in calculated_features]
        df = pd.DataFrame({
            'seed': pd.Series(img_seeds),
            'gan_embedding': pd.Series(used_embed),
            'predictions': pd.Series(pred_labels).astype(np.float32),
            'classifier_features': pd.Series(calc_feat).astype(object),
            'class_swap': pd.Series(swap_labels),
            'gan_labels': pd.Series(gan_labels),
            'categorical_pred': pd.Series(plot_labels),
            'linear_pred': pd.Series(pred_labels),
            'umap_x': pd.Series(umap[:, 0]),
            'umap_y': pd.Series(umap[:, 1])
        })
        df.to_parquet(f'{layer}_features.parquet.gzip', compression='gzip')
        df_diff = pd.DataFrame({

        })

    # Feature maps.
    sns.scatterplot(umap[:, 0], umap[:, 1], hue=gan_labels, s=7)
    plt.savefig(join(out, f'{layer}_gan_labels.svg'))
    plt.clf()
    sns.scatterplot(umap[:, 0], umap[:, 1], hue=plot_labels, s=7)
    plt.savefig(join(out, f'{layer}_class_preds.svg'))
    plt.clf()
    sns.scatterplot(umap[:, 0], umap[:, 1], hue=pred_labels, s=7)
    plt.savefig(join(out, f'{layer}_linear_preds.svg'))
    plt.clf()
    sns.scatterplot(umap[:, 0], umap[:, 1], hue=swap_labels, s=7)
    plt.savefig(join(out, f'{layer}_swap_labels.svg'))
    plt.clf()

    if lasso:
        utils.lasso_plot(umap[:, 0], umap[:, 1], out_path='selected_points_from_predict.txt', s=7)
        plt.show()

    # Write class-swap seeds.
    class_swap_path = join(out, 'class_swap_seeds.txt')
    with open(class_swap_path, 'w') as file:
        file.write('\n'.join([str(s) for s in class_swap_seeds]))
        print(f"Wrote class-swap seeds to {class_swap_path}")

    strong_class_swap_path = join(out, 'strong_class_swap_seeds.txt')
    with open(strong_class_swap_path, 'w') as file:
        file.write('\n'.join([str(s) for s in strong_class_swap_seeds]))
        print(f"Wrote class-swap seeds to {strong_class_swap_path}")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    predict() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
