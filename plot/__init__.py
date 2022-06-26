"""Plotting functions for class-conditional StyleGAN2."""

from typing import Any, List

import matplotlib.pyplot as plt
import torch
from PIL import Image

from .. import utils


def plot_embedding_images(
    E_G: torch.nn.Module,
    seeds: List[int],
    embed0: torch.tensor,
    embed1: torch.tensor,
    device: torch.device,
    gan_kwargs: Any = {},
    decode_kwargs: Any = {}
):
    scale = 5
    fig, ax = plt.subplots(len(seeds), 2, figsize=(2 * scale, len(seeds) * scale))
    for s, seed in enumerate(seeds):
        # Get the corresponding noise vector.
        z = utils.noise_tensor(seed, z_dim=E_G.z_dim).to(device)

         # First image (starting embedding, BRAF-like)
        img0 = E_G(z, embed0, **gan_kwargs)
        img0 = utils.process_gan_batch(img0)
        img0 = utils.decode_batch(img0, **decode_kwargs)
        img0 = Image.fromarray(img0['tile_image'].numpy()[0])

        # Second image (ending embedding, RAS-like)
        img1 = E_G(z, embed1, **gan_kwargs)
        img1 = utils.process_gan_batch(img1)
        img1 = utils.decode_batch(img1, **decode_kwargs)
        img1 = Image.fromarray(img1['tile_image'].numpy()[0])

        if len(seeds) == 1:
            _ax0 = ax[0]
            _ax1 = ax[1]
        else:
            _ax0 = ax[s, 0]
            _ax1 = ax[s, 1]

        if s == 0:
            _ax0.set_title('BRAF-like')
            _ax1.set_title('RAS-like')

        _ax0.imshow(img0)
        _ax1.imshow(img1)
        _ax0.axis('off')
        _ax1.axis('off')

    fig.subplots_adjust(wspace=0.05, hspace=0)

def plot_seeds_from_df(_df, idx):
    for i in idx:
        print("Seed {} Prediction: {:.3f} -> {:.3f}".format(
            _df.seed.values[i],
            _df.pred_start.values[i],
            _df.pred_end.values[i]
        ))
    plot_embedding_images(_df.seed.values[idx], embed0, embed1)
