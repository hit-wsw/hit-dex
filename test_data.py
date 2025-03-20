"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

import argparse
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import socket
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.macros as Macros
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings


def load_dict_from_checkpoint(ckpt_path):
    """
    Load checkpoint dictionary from a checkpoint file.

    Args:
        ckpt_path (str): Path to checkpoint file.

    Returns:
        ckpt_dict (dict): Loaded checkpoint dictionary.
    """
    ckpt_path = os.path.expandvars(os.path.expanduser(ckpt_path))
    if not torch.cuda.is_available():
        ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    else:
        ckpt_dict = torch.load(ckpt_path)
    return ckpt_dict


def maybe_dict_from_checkpoint(ckpt_path=None, ckpt_dict=None):
    """
    Utility function for the common use case where either an ckpt path
    or a ckpt_dict is provided. This is a no-op if ckpt_dict is not
    None, otherwise it loads the model dict from the ckpt path.

    Args:
        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

    Returns:
        ckpt_dict (dict): Loaded checkpoint dictionary.
    """
    assert (ckpt_path is not None) or (ckpt_dict is not None)
    if ckpt_dict is None:
        ckpt_dict = load_dict_from_checkpoint(ckpt_path)
    return ckpt_dict

def train(config, device, auto_remove_exp=False, resume=None):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # set num workers
    torch.set_num_threads(1)
    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config, auto_remove_exp_dir=auto_remove_exp)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    eval_dataset_cfg = config.train.data[0]
    dataset_path = os.path.expandvars(os.path.expanduser(eval_dataset_cfg["path"]))
    ds_format = config.train.data_format
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # # load basic metadata from training file
    # print("\n============= Loaded Environment Metadata =============")
    # env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)

    # # update env meta if applicable
    # from robomimic.utils.script_utils import deep_update
    # deep_update(env_meta, config.experiment.env_meta_update_dict)

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        ds_format=ds_format,
        verbose=True
    )

    print("==============================================\n")

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    print(shape_meta["all_shapes"])


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = [dict(path=args.dataset)]

    if args.name is not None:
        config.experiment.name = args.name

    if args.output is not None:
        config.train.output_dir = args.output

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        Macros.DEBUG = True

        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    important_stats = None

    important_stats = train(config, device=device, auto_remove_exp=args.auto_remove_exp, resume=args.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="(optional) checkpoint to load and resume training/validation",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # Output path, to override the one in the config
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="(optional) if provided, override the output folder path defined in the config",
    )

    # force delete the experiment folder if it exists
    parser.add_argument(
        "--auto-remove-exp",
        action='store_true',
        help="force delete the experiment folder if it exists"
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    args = parser.parse_args()
    main(args)

