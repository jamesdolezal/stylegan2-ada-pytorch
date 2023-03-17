# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""

import copy
import json
import os
import re
import tempfile

import slideflow as sf
import torch

from . import dnnlib
from .metrics import metric_main
from .torch_utils import custom_ops, training_stats
from .training import training_loop

#----------------------------------------------------------------------------

def load_project(sf_kwargs):
    dataset_kwargs = {k:v for k,v in sf_kwargs.items() if k in ('tile_px', 'tile_um', 'filters', 'filter_blank')}
    project = sf.Project(sf_kwargs['project_path'])
    dataset = project.dataset(**dataset_kwargs)
    return project, dataset

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def setup_training_loop_kwargs(
    # General options (not included in desc).
    gpus        = None, # Number of GPUs: <int>, default = 1 gpu
    snap        = None, # Snapshot interval: <int>, default = 50 ticks
    metrics     = 'fid50k_full', # List of metric names: [], ['fid50k_full'] (default), ...
    seed        = None, # Random seed: <int>, default = 0

    # Dataset.
    data        = None, # Training dataset (either this or `slideflow` is required): <path>
    slideflow   = None, # Slideflow configuration JSON (either this or `data` is required): <path>
    cond        = None, # Train conditional model based on dataset labels: <bool>, default = False
    subset      = None, # Train with only N images: <int>, default = all
    mirror      = None, # Augment dataset with x-flips: <bool>, default = False

    # Base config.
    cfg         = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
    gamma       = None, # Override R1 gamma: <float>
    kimg        = None, # Override training duration: <int>
    batch       = None, # Override batch size: <int>

    # Discriminator augmentation.
    aug         = None, # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
    p           = None, # Specify p for 'fixed' (required): <float>
    target      = None, # Override ADA target for 'ada': <float>, default = depends on aug
    augpipe     = None, # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'

    # Transfer learning.
    resume      = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
    freezed     = None, # Freeze-D: <int>, default = 0 discriminator layers

    # Performance options (not included in desc).
    fp32        = None, # Disable mixed-precision training: <bool>, default = False
    nhwc        = None, # Use NHWC memory format with FP16: <bool>, default = False
    allow_tf32  = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench     = None, # Disable cuDNN benchmarking: <bool>, default = False
    workers     = None, # Override number of DataLoader workers: <int>, default = 3
    lazy_resume = None, # Allow lazy loading from saved pretrained networks
):
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 50
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    if metrics is not None and not isinstance(metrics, list):
        metrics = [metrics]
    elif metrics is None:
        metrics = []
    assert isinstance(metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = metrics

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # -----------------------------------
    # Dataset: data, cond, subset, mirror
    # -----------------------------------

    assert (data is not None or slideflow is not None) and not (data is not None and slideflow is not None)
    assert data is None or isinstance(data, str)
    assert slideflow is None or isinstance(slideflow, str)
    interp_embed = False
    args.slideflow_kwargs = {}
    if data is not None:
        args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=cond, max_size=None, xflip=False)
    elif slideflow is not None:
        try:
            with open(slideflow, 'r') as sf_args_f:
                args.slideflow_kwargs = dnnlib.EasyDict(**json.load(sf_args_f))
        except IOError as err:
            raise UserError(f'--slideflow: {err}')
        if args.slideflow_kwargs.model_type not in ('categorical', 'linear'):
            raise UserError(f'Unknown slideflow model type {args.slideflow_kwargs.model_type}, must be "categorical" or "linear"')
        if args.slideflow_kwargs.model_type == 'linear':
            interp_embed = True
        project, dataset = load_project(args.slideflow_kwargs)
        outcome_key = 'outcomes' if 'outcomes' in args.slideflow_kwargs else 'outcome_label_headers'
        if args.slideflow_kwargs[outcome_key] is not None:
            labels, _ = dataset.labels(args.slideflow_kwargs[outcome_key], use_float=(args.slideflow_kwargs['model_type'] != 'categorical'))
        else:
            labels = None

        if 'loc_labels' in args.slideflow_kwargs:
            label_kwargs = dict(
                class_name='slideflow.io.torch.LocLabelInterleaver',
                loc_labels=args.slideflow_kwargs['loc_labels'],
                labels=None,
            )
        else:
            label_kwargs = dict(
                class_name='slideflow.io.torch.StyleGAN2Interleaver',
                labels=labels,
            )

        # Normalizer
        if 'normalizer_kwargs' in args.slideflow_kwargs:
            label_kwargs.update(args.slideflow_kwargs['normalizer_kwargs'])
            method = args.slideflow_kwargs['normalizer_kwargs']['normalizer']
            print(f"Using {method} normalization.")

        args.training_set_kwargs = dnnlib.EasyDict(tfrecords=dataset.tfrecords(),
                                                   img_size=args.slideflow_kwargs['tile_px'],
                                                   use_labels=cond,
                                                   chunk_size=4,
                                                   augment='xyr',
                                                   standardize=False,
                                                   num_tiles=dataset.num_tiles,
                                                   prob_weights=dataset.prob_weights,
                                                   model_type=args.slideflow_kwargs['model_type'],
                                                   onehot=True,
                                                   **label_kwargs)

    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    try:
        if slideflow is not None:
            args.training_set_kwargs.resolution = args.slideflow_kwargs.tile_px
            args.training_set_kwargs.use_labels = hasattr(args.slideflow_kwargs, 'outcome_label_headers') or hasattr(args.slideflow_kwargs, 'outcomes')
            args.training_set_kwargs.max_size = None
            with open(os.path.join(args.slideflow_kwargs.project_path, 'settings.json'), 'r') as sf_settings_f:
                desc = json.load(sf_settings_f)['name']
        else:
            training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs, **args.slideflow_kwargs) # subclass of training.dataset.Dataset
            args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
            args.training_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
            if data is not None:
                args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
            else:
                args.training_set_kwargs.max_size = None
            desc = training_set.name
            del training_set # conserve memory
    except IOError as err:
        raise UserError(f'--data: {err}')

    if cond is None:
        cond = False
    assert isinstance(cond, bool)
    if cond:
        if not args.training_set_kwargs.use_labels:
            raise UserError('--cond=True requires labels specified in dataset.json')
        desc += '-cond'
    else:
        args.training_set_kwargs.use_labels = False

    if subset is not None:
        assert isinstance(subset, int)
        if args.training_set_kwargs.max_size and not 1 <= subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{subset}'
        if args.training_set_kwargs.max_size and subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = subset
            args.training_set_kwargs.random_seed = args.random_seed

    if mirror is None:
        mirror = False
    assert isinstance(mirror, bool)
    if mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True
    else:
        args.training_set_kwargs.xflip = False

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        desc += f'{gpus:d}'
        spec.ref_gpus = gpus
        res = args.training_set_kwargs.resolution
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32

    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, interp_embed=interp_embed, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), interp_embed=interp_embed, mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if cfg == 'cifar':
        args.loss_kwargs.pl_weight = 0 # disable path length regularization
        args.loss_kwargs.style_mixing_prob = 0 # disable style mixing
        args.D_kwargs.architecture = 'orig' # disable residual skip connections

    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{gamma:g}'
        args.loss_kwargs.r1_gamma = gamma

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------

    if aug is None:
        aug = 'ada'
    else:
        assert isinstance(aug, str)
        desc += f'-{aug}'

    if aug == 'ada':
        args.ada_target = 0.6

    elif aug == 'noaug':
        pass

    elif aug == 'fixed':
        if p is None:
            raise UserError(f'--aug={aug} requires specifying --p')

    else:
        raise UserError(f'--aug={aug} not supported')

    if p is not None:
        assert isinstance(p, float)
        if aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc += f'-p{p:g}'
        args.augment_p = p

    if target is not None:
        assert isinstance(target, float)
        if aug != 'ada':
            raise UserError('--target can only be specified with --aug=ada')
        if not 0 <= target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{target:g}'
        args.ada_target = target

    assert augpipe is None or isinstance(augpipe, str)
    if augpipe is None:
        augpipe = 'bgc'
    else:
        if aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')
        desc += f'-{augpipe}'

    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

    assert augpipe in augpipe_specs
    if aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    elif resume in resume_specs:
        desc += f'-resume{resume}'
        args.resume_pkl = resume_specs[resume] # predefined url
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    if resume != 'noresume':
        args.ada_kimg = 100 # make ADA react faster at the beginning
        args.ema_rampup = None # disable EMA rampup

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if fp32 is None:
        fp32 = False
    assert isinstance(fp32, bool)
    if fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if nhwc is None:
        nhwc = False
    assert isinstance(nhwc, bool)
    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    assert isinstance(allow_tf32, bool)
    if allow_tf32:
        args.allow_tf32 = True

    if lazy_resume is None:
        lazy_resume = False
    assert isinstance(lazy_resume, bool)
    if lazy_resume:
        args.lazy_resume = True

    if workers is not None:
        assert isinstance(workers, int)
        if not workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = workers

    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **args) #G_reg_interval=None, D_reg_interval=None,

#----------------------------------------------------------------------------

def train(outdir, dry_run, ctx=None, **config_kwargs):
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError as e:
        sf.log.debug(
        f"Encountered runtime error attempting to set start method: {e}"
    )
    dnnlib.util.Logger(should_flush=True)

    # Setup training options.
    try:
        run_desc, args = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        if ctx is not None:
            ctx.fail(err)
        else:
            raise err

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)
    training_path = args.slideflow_kwargs.project_path if hasattr(args, 'slideflow_kwargs') else args.training_set_kwargs.path


    if hasattr(args, 'slideflow_kwargs'):
        args_for_print = copy.deepcopy(args)
        args_for_print.training_set_kwargs.tfrecords = '[...]'
        args_for_print.training_set_kwargs.labels = '[...]'
    else:
        args_for_print = args

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(args_for_print, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {training_path}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print(f'Number of images:   {args.training_set_kwargs.max_size}')
    print(f'Image resolution:   {args.training_set_kwargs.resolution}')
    print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
    print()

    if hasattr(args, 'slideflow_kwargs'):
        print('Slideflow options:')
        print(json.dumps(args.slideflow_kwargs, indent=2))
        print('Setting up TFRecord indices...')
        project, dataset = load_project(args.slideflow_kwargs)
        dataset.build_index(False)

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)
