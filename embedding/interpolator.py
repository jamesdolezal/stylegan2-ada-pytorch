import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import slideflow as sf
import tensorflow as tf
import torch
from tqdm import tqdm

from .. import embedding, plot, utils
from ..interpolate import class_interpolate, linear_interpolate


class Interpolator:

    def __init__(self, gan_pkl, device, start, end, **gan_kwargs):
        self.E_G, self.G = embedding.load_embedding_gan(gan_pkl, device)
        self.device = device
        self.gan_kwargs = gan_kwargs
        self.decode_kwargs = dict(standardize=False, resize_px=299)
        self.embed0, self.embed1 = embedding.get_class_embeddings(
            self.G,
            start=start,
            end=end,
            device=device
        )
        self.features = None
        self.normalizer = None

    def z(self, seed):
        return utils.noise_tensor(seed, self.E_G.z_dim).to(self.device)

    def set_feature_model(self, path, layers='postconv'):
        self.features = sf.model.Features(path, layers=layers, include_logits=True)
        self.normalizer = self.features.wsi_normalizer

    def seed_search(self, seeds, batch_size=32, verbose=False):
        if self.features is None:
            raise Exception("Feature model not set; use .set_feature_model()")
        return embedding.seed_search(
            seeds,
            self.embed0,
            self.embed1,
            self.E_G,
            self.features,
            self.device,
            batch_size,
            normalizer=self.normalizer,
            verbose=verbose,
            **self.gan_kwargs
        )

    def plot_comparison(self, seeds):
        if not isinstance(seeds, list):
            seeds = [seeds]
        plot.plot_embedding_images(
            self.E_G,
            seeds,
            self.embed0,
            self.embed1,
            self.device,
            gan_kwargs=self.gan_kwargs,
            decode_kwargs=self.decode_kwargs
        )

    def generate_tf_from_embedding(self, seed, embedding):
        z = self.z(seed)
        gan_out = self.E_G(z, embedding, **self.gan_kwargs)
        raw, processed = utils.process_gan_raw(
            gan_out,
            normalizer=self.normalizer,
            **self.decode_kwargs
        )
        return raw, processed


    def generate_tf_start(self, seed):
        return self.generate_tf_from_embedding(seed, self.embed0)

    def generate_tf_end(self, seed):
        return self.generate_tf_from_embedding(seed, self.embed1)

    def interpolate(self, seed, watch=None):
        imgs = []
        proc_imgs = []
        preds = []
        watch_out = []

        for img in tqdm(class_interpolate(self.G, self.z(seed), 0, 1, self.device, steps=100, **self.gan_kwargs), total=100):
            img = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0, 3, 1, 2)
            img = (img / 127.5) - 1
            img = utils.process_gan_batch(img)
            img = utils.decode_batch(img, **self.decode_kwargs)
            if self.normalizer:
                img = self.normalizer.batch_to_batch(img['tile_image'])[0]
            else:
                img = img['tile_image']
            processed_img = tf.image.per_image_standardization(img)
            img = img.numpy()[0]
            pred = self.features(processed_img)[-1].numpy()
            preds += [pred[0][0]]
            if watch is not None:
                watch_out += [watch(processed_img)]
            imgs += [img]
            proc_imgs += [processed_img[0]]

        sns.lineplot(x=range(len(preds)), y=preds, label="Prediction")
        plt.axhline(y=0, color='black', linestyle='--')
        plt.title("Prediction during interpolation")
        plt.xlabel("Interpolation Step (BRAF -> RAS)")
        plt.ylabel("Prediction")

        if watch is not None:
            return imgs, proc_imgs, preds, watch_out
        else:
            return imgs, proc_imgs, preds


