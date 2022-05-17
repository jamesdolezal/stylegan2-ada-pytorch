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
from os.path import exists, join
from typing import TYPE_CHECKING, List, Optional

import click
import numpy as np
import slideflow as sf
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer

#----------------------------------------------------------------------------

def seeds_from_dir(path: str) -> List[int]:
    return sorted([
        seed_from_path(f)
        for f in images_from_dir(path)
    ])


def seed_from_path(path: str) -> int:
    return int(sf.util.path_to_name(path)[-4:])


def images_from_dir(path: str) -> List[str]:
    return sorted([
        join(path, f) for f in os.listdir(path)
        if sf.util.path_to_ext(f).lower() in ('jpg', 'png')
    ])


def decode_img(
    img_path: str,
    normalizer: Optional["StainNormalizer"] = None
) -> np.ndarray:
    np_img = np.array(Image.open(img_path))
    if normalizer is not None:
        np_img = normalizer.rgb_to_rgb(np_img)
    np_img = np.expand_dims(np_img, axis=0)
    tf_img = tf.image.per_image_standardization(np_img)
    return {'tile_image': tf_img}

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--model', 'model_path', help='Path to classifier model', required=True)
@click.option('--path', type=str, help='Path to saved images for which predictions are generated', required=True, metavar='DIR')
@click.option('--backend', type=str, default='tensorflow', help='Backend for Slideflow classifier model.')
def predict(
    ctx: click.Context,
    model_path: str,
    path: str,
    backend: str
):
    """Generate predictions from saved images using a saved Slideflow classifier
    model. Seeds whose images have classifier predictions consistent with the
    GAN labels are marked with "*" and saved in the file `class_swap_seeds.txt`
    in the given directory. Seeds with predictions opposite of the GAN labels
    are marked with "(!)".

    This script currently only supports the Tensorflow backend and only supports
    binary categorical models.

    The path directory should contain subdirectories according to GAN labels:

    \b
    .../path/
        0/
            seed0001.jpg
            seed0002.jpg
            ...
        1/
            seed0001.jpg
            seed0002.jpg
            ...
    """

    if not exists(join(path, '0')) or not exists(join(path, '1')):
        ctx.fail("Unable to find class label directories at target path.")
    if backend not in ('torch', 'tensorflow'):
        ctx.fail("Unrecognized backend {}".format(backend))
    if backend == 'torch':
        ctx.fail("PyTorch backend not yet supported for this script.")

    if seeds_from_dir(join(path, '0')) != seeds_from_dir(join(path, '1')):
        raise ValueError("Mismatched seeds found in class directories.")

    num_seeds = len(seeds_from_dir(join(path, '0')))
    print("Detected {} seeds.".format(num_seeds))
    class_swap_seeds = []

    # Load model configuration.
    print('Reading model configuration...')
    config = sf.util.get_model_config(model_path)
    if config['hp']['normalizer']:
        normalizer = sf.norm.autoselect(config['hp']['normalizer'], config['hp']['normalizer_source'])
        if 'norm_fit' in config and config['norm_fit'] is not None:
            normalizer.fit(**config['norm_fit'])
    else:
        normalizer = None

    # Load model.
    print('Loading classifier from "%s"...' % model_path)
    model = tf.keras.models.load_model(model_path)

    # Generate predictions.
    for (img0, img1) in tqdm(zip(images_from_dir(join(path, '0')), images_from_dir(join(path, '1'))), total=num_seeds):
        seed = seed_from_path(img0)
        assert seed == seed_from_path(img1)
        pred0 = model.predict(decode_img(img0, normalizer=normalizer))[0][1]
        pred1 = model.predict(decode_img(img1, normalizer=normalizer))[0][1]
        if round(pred0) != round(pred1):
            if round(pred0):
                # Predictions are oppositve of what is expected.
                tail = " (!)"
            else:
                # Class-swapping is observed for this seed.
                tail = "*"
                class_swap_seeds += [seed]
        else:
            tail = ""
        tqdm.write(f"Seed {seed}: {pred0:.2f}\t{pred1:.2f}{tail}")

    # Write class-swap seeds.
    class_swap_path = join(path, 'class_swap_seeds.txt')
    with open(class_swap_path, 'w') as file:
        file.write('\n'.join([str(s) for s in class_swap_seeds]))
        print(f"Wrote class-swap seeds to {class_swap_path}")


#----------------------------------------------------------------------------

if __name__ == "__main__":
    predict() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
