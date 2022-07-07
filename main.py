import torch
import os
import random
import argparse
import logging
import numpy as np
from config import global_config as cfg

from GPTCritic import GPTCritic
from DecisionTransformer import DT
from WeightedBC import WeightedBC

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


def seed_initialize(seed):
    # fix random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_path(gpt_path, algorithm, iteration, seed):
    cfg.gpt_path = gpt_path
    cfg.exp_path = os.path.join('./experiments', '{}_iter_{}_seed_{}'.format(algorithm, iteration, seed))
    cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
    cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
    cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')
    cfg.eval_load_path = cfg.exp_path


def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-algorithm')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = args.mode
    if args.mode == 'test' or args.mode == 'adjust':
        parse_arg_cfg(args)
        cfg.eval_load_path = os.path.join('./experiments', '{}_iter_{}_seed_{}/best_model'.format(args.algorithm, cfg.iteration, cfg.seed))
        cfg.gpt_path = cfg.eval_load_path
    else:  # train
        parse_arg_cfg(args)
        if cfg.exp_path in ['', 'to be generated']:
            experiments_path = './experiments' if 'all' in cfg.exp_domains else './experiments_Xdomain'
            cfg.exp_path = os.path.join(experiments_path, '{}_iter_{}_seed_{}'.format(args.algorithm, cfg.iteration, cfg.seed))
            logging.info('save path:', cfg.exp_path)

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

    seed_initialize(cfg.seed)

    # initialize model
    if args.algorithm in ['GPT-Critic', 'UBAR']:
        m = GPTCritic(device)
    elif args.algorithm == 'DT':
        m = DT(device)
    elif args.algorithm == 'WBC':
        m = WeightedBC(device)
    else:
        raise ValueError("Incorrect algorithm name ", args.algorithm)

    if args.mode == 'train':    # train
        if args.algorithm == 'UBAR':
            m.bc(iteration=0, seed=cfg.seed)
        elif args.algorithm == 'GPT-Critic':
            # Fine-tuning the GPT-2 (initial policy)
            set_path('distilgpt2', args.algorithm, 0, cfg.seed)
            m = GPTCritic(device)
            m.bc(iteration=0, seed=cfg.seed)

            for iteration in range(cfg.iteration):
                cfg.exp_path = os.path.join('./experiments', '{}_iter_{}_seed_{}'.format(args.algorithm, iteration, cfg.seed))
                policy_path = os.path.join(cfg.exp_path, 'best_model')
                set_path(policy_path, args.algorithm, iteration, cfg.seed)
                m = GPTCritic(device)

                # Generate rewards for policy evaluation
                m.generate_rewards(iteration=iteration, seed=cfg.seed)

                # Policy evaluation
                m.policy_evaluation(iteration=iteration, seed=cfg.seed)

                # Self-generation with trained gpt-2 and critic
                m.self_generate(iteration=iteration, seed=cfg.seed)

                # Check the dataset performance
                m.dataset_evaluation(iteration=iteration, seed=cfg.seed)

                # Initialize seed
                seed_initialize(cfg.seed)

                # Policy update with fine-tuning the GPT-2 with updated dataset
                set_path('distilgpt2', args.algorithm, iteration+1, cfg.seed)
                m = GPTCritic(device)
                m.bc(iteration=iteration+1, seed=cfg.seed)
            m.generate_rewards(iteration=cfg.iteration, seed=cfg.seed)
            m.dataset_evaluation(iteration=cfg.iteration, seed=cfg.seed)
        elif args.algorithm == 'DT':
            m.generate_rewards()
            m.train()
        elif args.algorithm == 'WBC':
            m.generate_rewards()
            m.policy_evaluation(seed=cfg.seed)
            m.train()
        else:
            raise ValueError("Incorrect algorithm name ", args.algorithm)

    else:  # test
        # cfg.eval_load_path = os.path.join('./experiments', '{}_iter_{}_seed_{}/best_model'.format(args.algorithm, cfg.iteration, cfg.seed))
        logging.info("Generate setting: \n\t use true_prev_bspn={} \n\t use true_prev_aspn={} \n\t use true_db_pointer={} \n\t use true_prev_resp={} \n\t use true_curr_bspn={} \n\t use true_curr_aspn={} \n\t use_all_previous_context={}".format(
                            cfg.use_true_prev_bspn, cfg.use_true_prev_aspn, cfg.use_true_db_pointer, cfg.use_true_prev_resp,
                            cfg.use_true_curr_bspn, cfg.use_true_curr_aspn, cfg.use_all_previous_context
                        ))
        m.validate('test')

if __name__ == "__main__":
    main()

