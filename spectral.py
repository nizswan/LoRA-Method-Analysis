import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from timm.utils import NativeScaler, get_state_dict, ModelEma
from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from datasets import build_dataset
import ka_models
import utils
from spline_utils import set_spline_args


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--ckpt', default=None, help='Load checkpoint')
    # parser.add_argument('--arch', default='vit_small', type=str,
    #     choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    # parser.add_argument('--pretrained_weights', default='', type=str,
    #     help="Path to pretrained weights to load.")
    # parser.add_argument("--checkpoint_key", default="teacher", type=str,
    #     help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    # parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    # parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    # parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
    #     obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='vit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    # parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
    #                     help='Optimizer (default: "adamw"')
    # parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
    #                     help='Optimizer Epsilon (default: 1e-8)')
    # parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
    #                     help='Optimizer Betas (default: None, use opt default)')
    # parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
    #                     help='Clip gradient norm (default: None, no clipping)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                     help='SGD momentum (default: 0.9)')
    # parser.add_argument('--weight-decay', type=float, default=0.05,
    #                     help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    # parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
    #                     help='LR scheduler (default: "cosine"')
    # parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
    #                     help='learning rate (default: 5e-4)')
    # parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
    #                     help='learning rate noise on/off epoch percentages')
    # parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
    #                     help='learning rate noise limit percent (default: 0.67)')
    # parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
    #                     help='learning rate noise std-dev (default: 1.0)')
    # parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
    #                     help='warmup learning rate (default: 1e-6)')
    # parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    # parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
    #                     help='epoch interval to decay LR')
    # parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
    #                     help='epochs to warmup LR, if scheduler supports')
    # parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
    #                     help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    # parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
    #                     help='patience epochs for Plateau LR scheduler (default: 10')
    # parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
    #                     help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    # parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
    #                     help='Name of teacher model to train (default: "regnety_160"')
    # parser.add_argument('--teacher-path', type=str, default='')
    # parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    # parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    # parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    # parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    # parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10', 'CIFAR100', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    # parser.add_argument('--device', default='cuda',
    #                     help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    # parser.add_argument('--resume', default='', help='resume from checkpoint')
    # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    # parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    # parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    # parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    #########################################
    # Kolmogorov Arnold Attention Parameters
    parser.add_argument('--grid', default=10, type=int, help='Number of grids for the Spline')
    parser.add_argument('--order', default=10, type=int, help='Order of the Spline')
    parser.add_argument('--grid_range', default=[-4,4], type=int, nargs=2, help='Range of grids for the Spline')
    parser.add_argument('--base_fun', default='nn.SiLU', type=str, help='Base Function Custom: ZeroModule')
    parser.add_argument('--spline_type', default='BasisSpline', type=str, choices=['BasisSpline', 'BasisSpline_Eff', 'FourierSpline'], help='Type of Spline Function to use')
    parser.add_argument('--depth', default=2, type=int, help='Depth/No. of Layers')
    parser.add_argument('--hidden_dim', default=32, type=int, help='Hidden Dimension to use between input/output')
    parser.add_argument('--num_attention', default=None, type=int, help='Number of KA Attention for MixedHeads')
    parser.add_argument('--mixed_head', action='store_true', default=False, help='Enable Mixed Heads')
    parser.add_argument('--uni_head', action='store_true', default=False, help='Enable Single KAN Activation for each heads throughout the network')
    parser.add_argument('--hybrid_act', action='store_true', default=False, help='Enable MLP+KAN layered activation')
    parser.add_argument('--hybrid_mode', default='w1_phi2', type=str, choices=['w1_phi2', 'phi1_w2'], help='MLP+KAN Mode')
    parser.add_argument('--project_l1', action='store_true', default=False, help='Enable L1 Ball projection')
    parser.add_argument('--sp_not_trainable', action='store_false', default=True, help='Disable Spline Training')
    parser.add_argument('--sb_not_trainable', action='store_false', default=True, help='Disable Base Function Training')
    parser.add_argument('--alt_lr', type=float, default=None, help='Use a different LR for KAN Layers')
    #########################################
    # WandB Parameters
    parser.add_argument('--run_name', type=str, help='Name of Run to log with WandB')
    
    args = parser.parse_args()
    set_spline_args(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # if args.data_set == 'CIFAR10':
    #     args.nb_classes = 10
    # elif args.data_set == 'CIFAR100':
    #     args.nb_classes = 100
    # elif args.data_set == 'IMNET':
    #     args.nb_classes = 1000
    # else:
    #     raise NotImplementedError('Only CIFAR10, CIFAR100, ILSVRC are implemented')
    
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # if args.data_set == 'CIFAR10':
    #     args.nb_classes = 10
    # elif args.data_set == 'CIFAR100':
    #     args.nb_classes = 100
    # elif args.data_set == 'IMNET':
    #     args.nb_classes = 1000
    # else:
    #     raise NotImplementedError('Only CIFAR10, CIFAR100, ILSVRC are implemented')

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # build model
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )

    if args.ckpt:
        if args.ckpt.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.ckpt, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print('model loaded')

    # model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model.to(device)
    # if os.path.isfile(args.pretrained_weights):
    #     state_dict = torch.load(args.pretrained_weights, map_location="cpu")
    #     if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
    #         print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
    #         state_dict = state_dict[args.checkpoint_key]
    #     # remove `module.` prefix
    #     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    #     # remove `backbone.` prefix induced by multicrop wrapper
    #     state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    #     msg = model.load_state_dict(state_dict, strict=False)
    #     print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    # else:
    #     print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
    #     url = None
    #     if args.arch == "vit_small" and args.patch_size == 16:
    #         url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    #     elif args.arch == "vit_small" and args.patch_size == 8:
    #         url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
    #     elif args.arch == "vit_base" and args.patch_size == 16:
    #         url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    #     elif args.arch == "vit_base" and args.patch_size == 8:
    #         url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    #     if url is not None:
    #         print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
    #         state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    #         model.load_state_dict(state_dict, strict=True)
    #     else:
    #         print("There is no reference weights available for this model => We use random weights.")

    # open image
    use_dataset = False
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        # print("Since no image path have been provided, we take the first image in our paper.")
        # print("Since no image path have been provided, exiting the program.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        print('Using dataset')
        # sys.exit(1)
        use_dataset = True
        it = iter(data_loader_train)
        img, _ = next(it)
        img, _ = next(it)
        img, _ = next(it)
        img, _ = next(it)
        img, _ = next(it)
    transform = pth_transforms.Compose([
        pth_transforms.Resize((args.input_size,args.input_size)),
        pth_transforms.ToTensor(),
        # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        pth_transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    if not use_dataset: img = transform(img)

    # make the image divisible by the patch size
    # w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    # img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    if len(img.shape) == 3: img = img.unsqueeze(0)

    attentions, attentions_raw = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    # attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # if '_ka' in args.model or ('_lin' in args.model and '_lin_soft' not in args.model):
    #     attentions = (attentions - torch.min(attentions, dim=-1)[0])/(torch.max(attentions, dim=-1)[0] - torch.min(attentions, dim=-1)[0])

    attn_svdvals = []
    attn_raw_svdvals = []
    for h in range(nh):
        s = torch.linalg.svdvals(attentions[0,h,:,:])
        s_raw = torch.linalg.svdvals(attentions_raw[0,h,:,:])

        attn_svdvals.append(s)
        attn_raw_svdvals.append(s_raw)
    
    torch.save({'attn_svd':attn_svdvals, 'attn_raw_svd': attn_raw_svdvals, 'attn': attentions, 'attn_raw': attentions_raw}, os.path.join(args.output_dir, "spectral_data_5.pth"))