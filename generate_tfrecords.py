# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
from io import BytesIO
from typing import Optional

import click
import numpy as np
import PIL.Image
import slideflow as sf
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm

from stylegan2 import dnnlib, embedding, legacy

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--tiles', 'n_tiles', type=int, default=100, help='Number of tiles per tfrecord to generate')
@click.option('--tfrecords', 'n_tfrecords', type=int, default=1, help='Number of tfrecords to generate')
@click.option('--seed', type=int, default=0, help='Starting seed')
@click.option('--embed', type=bool, default=False, help='Generate images using middle of binary class embedding.')
@click.option('--name', type=str, default='gan', help='Name of tfrecord.')
@click.option('--format', type=str, help='TFRecord image format (PNG or JPEG)')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--gan_um', help='Tile size in microns of GAN images', type=int, default=400)
@click.option('--gan_px', help='Tile size in pixels of GAN images', type=int, default=400)
@click.option('--target_um', help='Tile size in microns of target images', type=int, default=400)
@click.option('--target_px', help='Tile size in pixels of target images', type=int, default=400)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    n_tfrecords: Optional[int],
    n_tiles: Optional[int],
    seed: Optional[int],
    embed: bool,
    name: Optional[str],
    format: str,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    gan_um: int,
    gan_px: int,
    target_um: int,
    target_px: int
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')

    os.makedirs(outdir, exist_ok=True)

    if embed:
        print("Generating images using middle embedding...")
        G, E_G = embedding.load_embedding_gan(network_pkl, device=device)
        embeddings = embedding.get_embeddings(G, device=device)
        embedding_first = embeddings[0].cpu().numpy()
        embedding_second = embeddings[1].cpu().numpy()
        interpolated_embedding = interp1d([0,2], np.vstack([embedding_first, embedding_second]), axis=0)
        m_embed = torch.from_numpy(np.expand_dims(interpolated_embedding(1), axis=0)).to(device)
    else:
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None and not embed:
            ctx.fail('Must specify class label with --class when using a conditional network')
        if not embed:
            label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    for tfr_idx in range(n_tfrecords):
        seeds = range((tfr_idx * n_tiles) + seed, (tfr_idx * n_tiles) + n_tiles + seed)

        # TFrecord writer.
        path = os.path.join(outdir, f"{name}{tfr_idx}.tfrecords")
        print(f"Writing to tfrecord at {path}")
        writer = sf.io.TFRecordWriter(path)

        # Generate images.
        for seed_idx, seed in tqdm(enumerate(seeds), desc=f"Generating {tfr_idx}/{50}...", total=len(seeds)):
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            if embed:
                img = E_G(z, m_embed, truncation_psi=truncation_psi, noise_mode=noise_mode)
            else:
                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            image = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

            # Resize/crop image.
            resize_factor = target_um / gan_um
            crop_width = int(resize_factor * gan_px)
            left = gan_px/2 - crop_width/2
            upper = gan_px/2 - crop_width/2
            right = left + crop_width
            lower = upper + crop_width
            image = image.crop((left, upper, right, lower)).resize((target_px, target_px))

            # Write to tfrecord.
            slidename_bytes = bytes(f'{name}{tfr_idx}', 'utf-8')
            with BytesIO() as output:
                image.save(output, format=format)
                record = sf.io.serialized_record(slidename_bytes, output.getvalue(), seed, 0)
            writer.write(record)

        writer.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
