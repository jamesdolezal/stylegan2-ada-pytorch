# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import imageio
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import interp1d

import dnnlib
import legacy
from training.networks import EmbeddingGenerator, EmbeddingMappingNetwork

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

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

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--start', type=int, help='Starting category for interpolation.')
@click.option('--end', type=int, help='Ending category for interpolation.')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--linear', help='Interpolate a linear outcome from 0-1', type=bool, metavar='BOOL')
@click.option('--video', help='Save in video (MP4) format. If false, will save side-by-side images.', default=True, show_default=True, type=bool, metavar='BOOL')
@click.option('--steps', help='Number of interpolation steps.', type=int, default=100, show_default=True)
def interpolate(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    start: Optional[int],
    end: Optional[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    linear: bool,
    video: bool,
    steps: int
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

    if steps < 2:
        ctx.fail("Steps must be greater than 1.")

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    if not linear:
        if start >= G.c_dim:
            raise ValueError(f"Starting index {start} too large, must be < {G.c_dim}")
        if end >= G.c_dim:
            raise ValueError(f"Ending index {end} too large, must be < {G.c_dim}")
        label_first = torch.zeros([1, G.c_dim], device=device)
        label_first[:, start] = 1
        label_second = torch.zeros([1, G.c_dim], device=device)
        label_second[:, end] = 1
        embedding_first = G.mapping.embed(label_first).cpu().numpy()
        embedding_second = G.mapping.embed(label_second).cpu().numpy()
        interpolated_embedding = interp1d([0, steps-1], np.vstack([embedding_first, embedding_second]), axis=0)
        G.mapping = EmbeddingMappingNetwork(G.mapping)
        E_G = EmbeddingGenerator(G)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        if video:
            video_file = imageio.get_writer(f'{outdir}/seed{seed:04d}.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')
            print (f'Saving optimization progress video "{outdir}/seed{seed:04d}.mp4"')
        else:
            out_img = Image.new('RGB', (299*steps, 299))
            x_offset = 0

        for interp_idx in range(steps):
            if linear:
                torch_interp = torch.tensor([[interp_idx/steps]]).to(device)
                img = G(z, torch_interp, truncation_psi=truncation_psi, noise_mode=noise_mode)
            else:
                embed = torch.from_numpy(np.expand_dims(interpolated_embedding(interp_idx), axis=0)).to(device)
                img = E_G(z, embed, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img + 1) * (255/2)
            img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            if video:
                video_file.append_data(img)
            else:
                out_img.paste(Image.fromarray(img), (x_offset, 0))
                x_offset += 299

        if video:
            video_file.close()
        else:
            out_img.save(f'{outdir}/seed{seed}.png')
#----------------------------------------------------------------------------

if __name__ == "__main__":
    interpolate() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
