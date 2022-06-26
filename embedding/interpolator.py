import slideflow as sf

from .. import embedding, plot, utils


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
        z = utils.noise_tensor(seed, self.E_G.z_dim).to(self.device)
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
