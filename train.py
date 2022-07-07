import copy
import pickle
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from eval import MultiWozEvaluator
from damd_net import DAMD, cuda_, get_one_hot_input
from reader import MultiWozReader
import utils
from torch.optim import Adam
import torch
import torch.nn as nn

import os
import shutil
import random
import argparse
import time
import logging
import json
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import global_config as cfg
# from config21 import global_config as cfg  # global, already initialized

import warnings
warnings.filterwarnings("ignore")


def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = args.mode
    if args.mode == 'test' or args.mode == 'adjust':
        parse_arg_cfg(args)
        cfg.gpt_path = cfg.eval_load_path
    else:  # train
        parse_arg_cfg(args)
        if cfg.exp_path in ['', 'to be generated']:
            experiments_path = './experiments' if 'all' in cfg.exp_domains else './experiments_Xdomain'
            cfg.exp_path = os.path.join(experiments_path, '{}_iter_{}_seed_{}'.format(cfg.exp_no, cfg.iteration, cfg.seed))
            # cfg.exp_path = os.path.join(experiments_path,'{}_{}_sd{}_lr{}_bs{}_ga{}'.format('-'.join(cfg.exp_domains),
            #                                                               cfg.exp_no, cfg.seed, cfg.lr, cfg.batch_size,
            #                                                               cfg.gradient_accumulation_steps))
            logging.info('save path:', cfg.exp_path)
            if cfg.save_log:
                if not os.path.exists(cfg.exp_path):
                    os.mkdir(cfg.exp_path)

            # to gpt later
            cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
            cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')
            cfg.eval_load_path = cfg.exp_path

    cfg._init_logging_handler(args.mode)
    if cfg.cuda:
        if len(cfg.cuda_device) == 1:
            cfg.multi_gpu = False
            device = torch.device("cuda:{}".format(cfg.gpu_idx) if torch.cuda.is_available() else "cpu")
        else:
            pass  # multi-gpu
    else:
        device = torch.device('cpu')
        logging.info('Device: {}'.format(torch.cuda.current_device()))

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # initialize model
    m = Modal(device)

    if args.mode == 'train':    # train
        if cfg.save_log:  # save cfg details.
            pass
        if cfg.context_scheme == 'UBARU':
            # m.find_example()
            ### 0. Reorder the dialogue state ###
            # m.relax_dialog_state()

            ### 1. Fine-tuning GPT-2 (UBAR) ###
            # m.bc(iteration=cfg.iteration, seed=cfg.seed)

            ### 2. Generate reward data for policy evaluation ###
            # m.generate_rewards(iteration=cfg.iteration, seed=cfg.seed)

            ### 3. Policy evaluation ###
            # m.policy_evaluation(iteration=cfg.iteration, seed=cfg.seed)

            ### 4. Self-generation ###
            m.self_generate(iteration=cfg.iteration, seed=cfg.seed)

            ### Check the dataset performance ###
            # m.dataset_evaluation()
        else:
            logging.info('Invalid context Scheme. must be UBARU or URURU')
            exit()
    else:  # test
        logging.info("Generate setting: \n\t use true_prev_bspn={} \n\t use true_prev_aspn={} \n\t use true_db_pointer={} \n\t use true_prev_resp={} \n\t use true_curr_bspn={} \n\t use true_curr_aspn={} \n\t use_all_previous_context={}".format(
                            cfg.use_true_prev_bspn, cfg.use_true_prev_aspn, cfg.use_true_db_pointer, cfg.use_true_prev_resp,
                            cfg.use_true_curr_bspn, cfg.use_true_curr_aspn, cfg.use_all_previous_context
                        ))

        if cfg.context_scheme == 'UBARU':
            m.validate('test')

if __name__ == "__main__":
    main()
