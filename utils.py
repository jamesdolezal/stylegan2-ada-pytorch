from functools import partial
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pyvips
import slideflow as sf
import tensorflow as tf
import torch
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from PIL import Image
from torchvision import transforms

if TYPE_CHECKING:
    from slideflow.norm import StainNormalizer


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def lasso_plot(x, y, out_path=None, **kwargs):
    # Create plot.
    fig, ax = plt.subplots()
    pts = ax.scatter(x, y, **kwargs)
    selector = SelectFromCollection(ax, pts)

    def accept(event):
        if event.key == "enter":
            if out_path:
                with open(out_path, 'w') as f:
                    f.write('\n'.join([str(i) for i in selector.ind]))
                print("Wrote {} points to {}.".format(len(selector.ind), out_path))
            else:
                print("Selected {} points:".format(len(selector.ind)))
                print(selector.ind)
            selector.disconnect()
            fig.canvas.draw()

    # Create lasso selector.
    fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept selected points.")


def decode_img(
    img: np.ndarray,
    normalizer: Optional["StainNormalizer"] = None
) -> np.ndarray:
    if normalizer is not None:
        img = normalizer.rgb_to_rgb(img)
    img = np.expand_dims(img, axis=0)
    tf_img = tf.image.per_image_standardization(img)
    return {'tile_image': tf_img}


def vips_resize(img, crop_width, target_px):
    img_data = np.ascontiguousarray(img.numpy()).data
    vips_image = pyvips.Image.new_from_memory(img_data, crop_width, crop_width, bands=3, format="uchar")
    vips_image = vips_image.resize(target_px/crop_width)
    img = sf.slide.vips2numpy(vips_image)
    return img


@tf.function
def decode_batch(
    img,
    normalizer: Optional["StainNormalizer"] = None
) -> np.ndarray:
    if normalizer is not None:
        img = normalizer.batch_to_batch(img)[0]
    tf_img = tf.image.per_image_standardization(img)
    return {'tile_image': tf_img}


def process_gan_image(img: torch.Tensor, normalizer=None):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = Image.fromarray(img[0].cpu().numpy(), 'RGB')

    # Resize/crop.
    gan_um = 400
    gan_px = 512
    target_um = 302
    target_px = 299
    resize_factor = target_um / gan_um
    crop_width = int(resize_factor * gan_px)
    left = gan_px/2 - crop_width/2
    upper = gan_px/2 - crop_width/2
    right = left + crop_width
    lower = upper + crop_width
    img = img.crop((left, upper, right, lower))
    img = img.resize((target_px, target_px))
    image = decode_img(img, normalizer=normalizer)
    return image


def process_gan_batch(img: torch.Tensor, normalizer=None, resize_method='tf_aa'):

    if (resize_method is not None
       and resize_method not in ('tf', 'tf_aa', 'torch', 'torch_aa', 'vips')):
        raise ValueError(f'Invalid resize method {resize_method}')

    # Calculate parameters for resize/crop.
    gan_um = 400
    gan_px = 512
    target_um = 302
    target_px = 299
    resize_factor = target_um / gan_um
    crop_width = int(resize_factor * gan_px)
    left = int(gan_px/2 - crop_width/2)
    upper = int(gan_px/2 - crop_width/2)

    # Perform crop/resize and convert to tensor
    img = transforms.functional.crop(img, upper, left, crop_width, crop_width)

    # Resize with PyTorch
    if resize_method in ('torch', 'torch_aa'):
        img = transforms.functional.resize(img, (target_px, target_px), antialias=(resize_method=='torch_aa'))

    # Re-order the dimension from BCWH -> BWHC
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

    # Resize with VIPS
    if resize_method == 'vips':
        img = [vips_resize(i, crop_width=crop_width, target_px=target_px) for i in img]

    # Convert to Tensorflow tensor
    img = tf.convert_to_tensor(img)

    # Resize with Tensorflow
    if resize_method in ('tf', 'tf_aa'):
        img = tf.image.resize(img, (target_px, target_px), method='lanczos3', antialias=(resize_method=='tf_aa'))

    # Normalize and standardize the image
    image = decode_batch(img, normalizer=normalizer)

    return image
