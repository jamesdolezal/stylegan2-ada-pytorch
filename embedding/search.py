from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import slideflow as sf
import tensorflow as tf
import torch
import utils
from sklearn.decomposition import PCA
from slideflow.util import colors as col
from slideflow.util import log
from tqdm.auto import tqdm


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
        z = utils.noise_tensor(seed, self.E_G.z_dim)[0].to(self.device)

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

        z = utils.noise_tensor(seed, z_dim=self.E_G.z_dim)[0].to(self.device)
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
