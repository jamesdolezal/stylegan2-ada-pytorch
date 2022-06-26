"""Embedding GAN functions."""

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from tqdm import tqdm
from training.networks import EmbeddingGenerator, EmbeddingMappingNetwork

from .. import dnnlib, legacy, utils
from .interpolator import Interpolator

if TYPE_CHECKING:
    import slideflow as sf
    import tensorflow as tf


def load_embedding_gan(
    gan_pkl: str,
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    print('Loading networks from "%s"...' % gan_pkl)
    with dnnlib.util.open_url(gan_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema']
        if device is not None:
            G = G.to(device) # type: ignore
    G.mapping = EmbeddingMappingNetwork(G.mapping)
    return EmbeddingGenerator(G), G


def get_class_embeddings(
    G: torch.nn.Module,
    start: int,
    end: int,
    device: Optional[torch.device] = None
):
    label_first = torch.zeros([1, G.c_dim], device=device)
    label_first[:, start] = 1
    label_second = torch.zeros([1, G.c_dim], device=device)
    label_second[:, end] = 1
    embedding_first = G.mapping.embed(label_first).cpu().numpy()
    embedding_second = G.mapping.embed(label_second).cpu().numpy()
    embed0 = torch.from_numpy(embedding_first)
    embed1 = torch.from_numpy(embedding_second)
    if device is not None:
        embed0 = embed0.to(device)
        embed1 = embed1.to(device)
    return embed0, embed1


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
    E_G, G = load_embedding_gan(gan_pkl, device=device)
    embed0, embed1 = get_class_embeddings(G, start, end, device=device)
    return E_G, embed0, embed1


def seed_search(
    seeds: List[int],
    embed0: torch.tensor,
    embed1: torch.tensor,
    E_G: torch.nn.Module,
    classifier_features: Union["tf.keras.models.Model", torch.nn.Module],
    device: torch.device,
    batch_size: int = 32,
    normalizer: Optional["sf.norm.StainNormalizer"] = None,
    verbose: bool = True,
    **gan_kwargs
):

    predictions = {0: [], 1: []}
    features = {0: [], 1: []}
    swap_labels = []
    img_seeds = []

    # --- GAN-Classifier pipeline ---------------------------------------------
    def gan_generator(embedding):
        def generator():
            for seed_batch in utils.batch(seeds, batch_size):
                z = torch.stack([utils.noise_tensor(s, z_dim=E_G.z_dim)[0] for s in seed_batch]).to(device)
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
    for (seed_batch, embed0_batch, embed1_batch) in zip(utils.batch(seeds, batch_size), gan_embed0_dts, gan_embed1_dts):

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
    return df
