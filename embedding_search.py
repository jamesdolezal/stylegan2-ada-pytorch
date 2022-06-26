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

from .. import embedding

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
    E_G, embed0, embed1 = embedding.load_gan_and_embeddings(gan_pkl, start, end, device)

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

    # Perform seed search
    df = embedding.seed_search(
        seeds,
        embed0,
        embed1,
        E_G,
        classifier_features,
        device,
        batch_size,
        normalizer,
        verbose,
        **gan_kwargs
    )

    features0 = np.stack(df.features_start.values)
    features1 = np.stack(df.features_end.values)
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
        pca.fit(np.concatenate([features0, features1]))
        all_pc_diff = pca.transform(features1) - pca.transform(features0)
    elif pca_method == 'delta':
        # Fit to strong-swap features only
        ss_features_start = np.stack(strong_swap_df.features_start.to_numpy())
        ss_features_end = np.stack(strong_swap_df.features_end.to_numpy())
        pca.fit(ss_features_end - ss_features_start)
        all_pc_diff = pca.transform(np.array(features1) - np.array(features0))
    for (xpc, ypc) in [[0, n] for n in range(1, pca_dim)]:
        plt.clf()
        sns.scatterplot(
            x=all_pc_diff[:, xpc],
            y=all_pc_diff[:, ypc],
            hue=(np.array(df.pred_end.values) - np.array(df.pred_start.values)),
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
        ES = embedding.search.EmbeddingSearch(
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
