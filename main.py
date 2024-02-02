"""
@File        :main.py
@Author      :Xinyu Lu
@EMail       :xinyulu@stu.xmu.edu.cn
"""

import os
import time
import json
import logging
import argparse
import config
from utils.utils import seed_everything, load_net_state, train_model, test_model


def get_args_parser():
    parser = argparse.ArgumentParser('Patch-based Auto Encoder for Classification (PACE)', add_help=False)

    # basic params
    parser.add_argument('--net', default='PACE',
                        help="Choose network")
    parser.add_argument('--ds', default='COV',
                        help="Choose dataset")
    parser.add_argument('--device', default='cuda:0',
                        help="Choose GPU device")

    parser.add_argument('-train', '--train', action='store_true',
                        help="start train")
    parser.add_argument('-tune', '--tune', action='store_true',
                        help="start tune")
    parser.add_argument('-test', '--test', action='store_true',
                        help="start test")

    parser.add_argument('--base_checkpoint',
                        help="Choose base model for fine-tune")
    parser.add_argument('--test_checkpoint',
                        help="Choose checkpoint for test")
    parser.add_argument('--seed',
                        default=2024,
                        help="Random seed")

    # params of PACE
    parser.add_argument('--dim',
                        help="dim of net")
    parser.add_argument('--depth',
                        help="depth of net")
    parser.add_argument('--kernel_size',
                        help="kernel size of net")
    parser.add_argument('--patch_size',
                        help="patch size of net")
    parser.add_argument('--pool_dim',
                        help="pool dim of net")

    # params of strategy
    parser.add_argument('--train_size',
                        help="train size for train_val_split")
    parser.add_argument('--batch_size',
                        help="batch size for training")
    parser.add_argument('--epoch',
                        help="epochs for training")
    parser.add_argument('--lr',
                        help="learning rate")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args_parser()

    if args.train and args.ds == 'Bacteria':
        args.lr = '1e-3'
        args.epoch = '50'

    if args.tune and args.ds == 'Bacteria':
        args.batch_size = '8'

    seed_everything(int(args.seed))
    ts = time.strftime('%Y-%m-%d_%H_%M', time.localtime())

    ds = args.ds
    net_ = args.net
    device = args.device

    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists(f'logs/{ds}'):
        os.mkdir(f'logs/{ds}')
    if not os.path.exists(f'logs/{ds}/{net_}'):
        os.mkdir(f'logs/{ds}/{net_}')

    if args.train:
        mode = "train"
    elif args.tune:
        mode = "tune"
    elif args.test:
        mode = "test"

    logging.basicConfig(
        filename=f'logs/{ds}/{net_}/{ts}_{mode}.log',
        format='%(levelname)s:%(message)s',
        level=logging.INFO,
    )

    logging.info({k: v for k, v in args.__dict__.items() if v})

    # ================================2. data & params ======================================
    data_root = f"datasets/{ds}"
    save_path = f"checkpoints/{ds}/{net_}/{ts}"

    if args.train:
        if not os.path.exists(f"checkpoints/{ds}"):
            os.mkdir(f"checkpoints/{ds}")
        if not os.path.exists(f"checkpoints/{ds}/{net_}"):
            os.mkdir(f"checkpoints/{ds}/{net_}")
        if not os.path.exists(f"checkpoints/{ds}/{net_}/{ts}"):
            os.mkdir(f"checkpoints/{ds}/{net_}/{ts}")

    if args.ds == 'FunctionalGroups':
        n_classes = 17
    else:
        n_classes = len(json.load(open(os.path.join(data_root, 'label.json'))))

    params = {'net': config.NET, 'strategy': config.STRATEGY['train'] if args.train else config.STRATEGY['tune']}

    if args.dim:
        params['net']['dim'] = int(args.dim)
    if args.depth:
        params['net']['depth'] = int(args.depth)
    if args.kernel_size:
        params['net']['kernel_size'] = int(args.kernel_size)
    if args.patch_size:
        params['net']['patch_size'] = int(args.patch_size)
    if args.pool_dim:
        params['net']['pool_dim'] = int(args.pool_dim)
    if args.batch_size:
        params['strategy']['batch_size'] = int(args.batch_size)
    if args.epoch:
        params['strategy']['epoch'] = int(args.epoch)
    if args.lr:
        params['strategy']['Adam_params']["lr"] = float(args.lr)
    if args.train_size:
        params['strategy']['train_size'] = float(args.train_size)

    if net_ == 'PACE':
        from models.PACE import PACE

        net = PACE(n_classes=n_classes, **params['net']).to(device)

    # ================================3. start to train/tune/test ======================================
    try:

        if args.train:
            train_model(net, save_path, ds=args.ds, device=device, **params['strategy'])

        elif args.tune:
            import torch

            base_model_path = args.base_model_path
            print(base_model_path)
            net = load_net_state(net, torch.load(f'{base_model_path}.pth'))
            train_model(net, save_path=f"{base_model_path}/tune", ds=args.ds, device=device,
                        tune=True, **params['strategy'])

        elif args.test:
            import torch

            test_model_path = args.test_checkpoint
            print(test_model_path)
            net = load_net_state(net, torch.load(test_model_path,
                                                 map_location={'cuda:0': device, 'cuda:1': device}))
            test_model(net, device=device, ds=args.ds)

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(e)
        os.remove(f'logs/{ds}/{net_}/{ts}_{mode}.log')
