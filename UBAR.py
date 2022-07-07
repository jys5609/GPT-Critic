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


class UBAR(object):
    def __init__(self, device):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path)
        self.reader = MultiWozReader(self.tokenizer)
        self.model = GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
        self.gamma = 0.99

        if cfg.mode == 'train':
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)  # single gpu

        self.evaluator = MultiWozEvaluator(self.reader)
        if cfg.save_log and cfg.mode == 'train':
            self.tb_writer = SummaryWriter(log_dir='./log')
        else:
            self.tb_writer = None

    def get_optimizers(self):
        """
        Setup the optimizer and the learning rate scheduler.

        from transformers.Trainer

        parameters from cfg: lr (1e-3); warmup_steps
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        num_training_steps = self.reader.set_stats['train']['num_dials'] * \
                             cfg.epoch_num // (cfg.gradient_accumulation_steps * cfg.batch_size)
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps * 0.2)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        critic_optimizer = AdamW(self.critic.parameters(), lr=cfg.lr)
        return optimizer, scheduler, critic_optimizer

    def log_first_inputs(self, inputs):
        tokenizer = self.tokenizer
        logging.info("**** Input Examples: ****")
        for context in inputs['contexts'][:4]:
            # ubar = tokenizer.convert_ids_to_tokens(context)
            # ubar = tokenizer.convert_tokens_to_string(context)
            # ubar = " ".join(ubar)
            ubar = tokenizer.decode(context)
            logging.info(ubar)

    def add_torch_transition_input(self, inputs):
        # to tensor and to device

        ### Batch data ###
        sa_tensor = torch.from_numpy(inputs['sa_np']).long()
        sa_tensor = sa_tensor.to(self.device)
        inputs['sa_tensor'] = sa_tensor

        next_sa_tensor = torch.from_numpy(inputs['next_sa_np']).long()
        next_sa_tensor = next_sa_tensor.to(self.device)
        inputs['next_sa_tensor'] = next_sa_tensor

        next_s_tensor = torch.from_numpy(inputs['next_s_np']).long()
        next_s_tensor = next_s_tensor.to(self.device)
        inputs['next_s_tensor'] = next_s_tensor

        sa_indices = inputs['eos_sa']
        next_sa_indices = inputs['eos_next_sa']
        next_s_indices = inputs['eos_next_s']

        mask_sa = np.zeros(inputs['sa_np'].shape)
        mask_next_sa = np.zeros(inputs['next_sa_np'].shape)
        mask_next_s = np.zeros(inputs['next_s_np'].shape)

        for idx, sa_idx in enumerate(sa_indices):
            mask_sa[idx][sa_idx] = 1.0
        for idx, next_sa_idx in enumerate(next_sa_indices):
            mask_next_sa[idx][next_sa_idx] = 1.0
        for idx, next_s_idx in enumerate(next_s_indices):
            mask_next_s[idx][next_s_idx] = 1.0

        mask_sa_tensor = torch.from_numpy(mask_sa).long()
        mask_sa_tensor = mask_sa_tensor.to(self.device)
        inputs['mask_sa_tensor'] = mask_sa_tensor.unsqueeze(-1)

        mask_next_sa_tensor = torch.from_numpy(mask_next_sa).long()
        mask_next_sa_tensor = mask_next_sa_tensor.to(self.device)
        inputs['mask_next_sa_tensor'] = mask_next_sa_tensor.unsqueeze(-1)

        mask_next_s_tensor = torch.from_numpy(mask_next_s).long()
        mask_next_s_tensor = mask_next_s_tensor.to(self.device)
        inputs['mask_next_s_tensor'] = mask_next_s_tensor.unsqueeze(-1)

        terminal_mask = np.zeros(inputs['next_sa_np'].shape)
        for idx, terminal in enumerate(inputs['terminal']):
            terminal_mask[idx][:] = 1 - terminal
        terminal_mask_tensor = torch.from_numpy(terminal_mask).long()
        terminal_mask_tensor = terminal_mask_tensor.to(self.device)
        inputs['terminal_mask_tensor'] = terminal_mask_tensor.unsqueeze(-1)

        reward = np.zeros(inputs['next_sa_np'].shape)
        for idx, r in enumerate(inputs['r']):
            # reward[idx][:] = r
            reward[idx][0] = r
        reward_tensor = torch.from_numpy(reward).long()
        reward_tensor = reward_tensor.to(self.device)
        inputs['reward_tensor'] = reward_tensor.unsqueeze(-1)
        return inputs

    def add_torch_input(self, inputs):
        # to tensor and to device
        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        contexts_tensor = contexts_tensor.to(self.device)
        inputs['contexts_tensor'] = contexts_tensor

        maskings_tensor = torch.from_numpy(inputs['maskings_np']).long()
        maskings_tensor = maskings_tensor.to(self.device)
        inputs['maskings_tensor'] = maskings_tensor
        return inputs

    def add_torch_input_eval(self, inputs):
        # inputs: context
        inputs['context_tensor'] = torch.tensor(
            [inputs['context']]).to(self.device)
        return inputs

    def calculate_loss_with_weight(self, outputs, labels, weight):
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        pad_id = cfg.pad_id
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')

        loss = None

        batch_size = outputs[0].shape[0]
        for i in range(batch_size):
            if loss is None:
                loss = loss_fct(shift_logits[i], shift_labels[i]) * weight[i]
            else:
                loss += loss_fct(shift_logits[i], shift_labels[i]) * weight[i]

        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss

    def calculate_loss_and_accuracy(self, outputs, labels, advantage_weights=None):
        if advantage_weights == None:
            lm_logits = outputs[0]

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            pad_id = cfg.pad_id

            loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            not_ignore = shift_labels.ne(pad_id)
            num_targets = not_ignore.long().sum().item()

            loss /= num_targets

        else:
            lm_logits = outputs[0]

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_advantage_weights = advantage_weights[..., 1:].contiguous()

            pad_id = cfg.pad_id

            loss_fct = nn.CrossEntropyLoss(size_average=False, ignore_index=pad_id, reduce=False, reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)) * shift_advantage_weights.view(-1)

            # avg loss
            not_ignore = shift_labels.ne(pad_id)
            num_targets = not_ignore.long().sum().item()

            loss = torch.sum(loss)
            loss /= num_targets
        return loss

    def dataset_evaluation(self):
        iteration = [3]
        seed = [0, 1, 2]

        for iter in iteration:
            for s in seed:
                successes = np.load('./data/scores/cum_successes_iter_{}_seed_{}.npy'.format(iter, s),
                                    allow_pickle=True)
                matches = np.load('./data/scores/cum_matches_iter_{}_seed_{}.npy'.format(iter, s), allow_pickle=True)

                print('Iteration: {} / Seed: {}'.format(iter, s))
                print('Success: {:.4f} / Match: {:.4f}'.format(np.mean(successes), np.mean(matches)))
            print()

    def relax_dialog_state(self):
        count = 0
        domain_miss_count = 0
        total_count = 0

        for data in self.reader.train:
            domains = []
            total_count += 1
            check = False
            domain_miss_check = False
            for turn in data:
                for turn_domain in turn['turn_domain']:
                    if (not turn_domain in domains) and turn_domain != '[general]':
                        domains.append(turn_domain)

                if len(domains) > 1:
                    bs = self.tokenizer.decode(turn['bspn'])

                    indices = []
                    encoded_indices = []
                    for domain in domains:
                        if not domain in bs:
                            domain_miss_check = True
                        else:
                            indices.append(bs.index(domain))
                            encoded_indices.append(turn['bspn'].index(self.tokenizer.encode(domain)[0]))
                    # check = False
                    for i in range(len(indices) - 1):
                        if indices[i] < indices[i + 1]:
                            check = True
                    if check:
                        sorted_indices = copy.deepcopy(encoded_indices)
                        sorted_indices.sort()
                        sorted_indices.append(-1)

                        relaxed_bs = [self.tokenizer.encode('<sos_b>')[0]]

                        for dom in domains[::-1]:
                            if not self.tokenizer.encode(dom)[0] in turn['bspn']:
                                continue
                            start_idx = turn['bspn'].index(self.tokenizer.encode(dom)[0])
                            sort_idx = sorted_indices.index(start_idx)
                            end_idx = sorted_indices[sort_idx + 1]
                            relaxed_bs.extend(turn['bspn'][start_idx:end_idx])

                        relaxed_bs.append(self.tokenizer.encode('<eos_b>')[0])
                        turn['bspn'] = relaxed_bs

            if domain_miss_check:
                domain_miss_count += 1

            if check:
                count += 1

        with open('data/self-generation/relaxed_train.pickle', 'wb') as f:
            pickle.dump(self.reader.train, f, pickle.HIGHEST_PROTOCOL)

    def find_example(self):
        with open('data/self-generation/relaxed_train.pickle', 'rb') as f:
            original_data = pickle.load(f)

        with open('data/self-generation-backup_0910/relaxed_train_iter_2.pickle', 'rb') as f:
            updated_data = pickle.load(f)

        for i in range(len(original_data)):
            # print('DATA NUM: {}'.format(i))
            _, _, cum_scores = self.evaluator.training_data_eval(original_data[i:i + 1])
            _, _, gen_cum_scores = self.evaluator.training_data_eval(updated_data[i:i + 1])

            for j in range(len(cum_scores)):
                if cum_scores[j] < gen_cum_scores[j]:
                    print('###################### USER GOAL #######################')
                    print(original_data[i][j]['dial_id'])
                    goal = self.reader.data[original_data[i][j]['dial_id']]['goal']
                    for domain in list(goal.keys()):
                        print(domain)
                        print(goal[domain])
                        print()

                    print('########## TURN: {} #########'.format(j))
                    print('### ORIGINAL ###')
                    for type in ['user', 'bspn', 'db', 'aspn', 'resp']:
                        print(type)
                        print(self.tokenizer.decode(original_data[i][j][type]))

                    print('### UPDATED ###')
                    for type in ['user', 'bspn', 'db', 'aspn', 'resp']:
                        print(type)
                        print(self.tokenizer.decode(updated_data[i][j][type]))

                    print('############################################################')

    def bc(self, iteration=0, seed=0):
        """
        Behavior cloning with revised dataset
        """

        if iteration == 0:
            with open('data/self-generation/train_iter_{}.pickle'.format(iteration), 'rb') as f:
                self.reader.train = pickle.load(f)
        else:
            with open('data/self-generation/train_iter_{}_seed_{}.pickle'.format(iteration, seed), 'rb') as f:
                self.reader.train = pickle.load(f)

        all_batches = self.reader.get_batches('train')
        valid_all_batches = self.reader.get_batches('dev')
        optimizer, scheduler, _ = self.get_optimizers()

        # log info
        set_stats = self.reader.set_stats['train']
        logging.info("***** Running training *****")
        logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                     set_stats['num_training_steps_per_epoch'])
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d",
                     set_stats['num_dials'] * cfg.epoch_num // (cfg.gradient_accumulation_steps * cfg.batch_size))

        # tb writer
        if self.tb_writer is not None:
            self.tb_writer.add_text('cfg', json.dumps(cfg.__dict__, indent=2))

        log_inputs = 2
        global_step = 0
        best_score = 0
        valid_best_loss = 1e10
        prev_path = ''

        valid_losses = []

        for epoch in range(cfg.epoch_num):
            if len(valid_losses) > 10 and all(v > valid_best_loss for v in valid_losses[-20:]):
                print('************ EARLY STOPPING ************')
                break

            epoch_step = 0
            tr_loss = 0.0
            logging_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()

            data_iterator = self.reader.get_nontranspose_data_iterator(all_batches)

            for batch_idx, dial_batch in enumerate(data_iterator):
                inputs = self.reader.convert_batch_session(dial_batch)
                try:  # avoid OOM
                    self.model.train()
                    if log_inputs > 0:  # log inputs for the very first two turns
                        self.log_first_inputs(inputs)
                        log_inputs -= 1

                    inputs = self.add_torch_input(inputs)
                    outputs = self.model(inputs['contexts_tensor'])
                    loss = self.calculate_loss_and_accuracy(outputs, labels=inputs['contexts_tensor'])
                    loss.backward()
                    tr_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    epoch_step += 1

                    # step, wrt gradient_accumulation_steps, clip grad norm
                    if (epoch_step + 1) % cfg.gradient_accumulation_steps == 0 or \
                            ((epoch_step + 1) == set_stats['num_training_steps_per_epoch']):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                        logs = {}  # for tb writer
                        # logging: loss, lr... after certain amount of steps
                        if cfg.report_interval > 0 and global_step % cfg.report_interval == 0:
                            loss_scalar = (tr_loss - logging_loss) / \
                                          cfg.report_interval
                            logging_loss = tr_loss
                            logs['loss'] = loss_scalar
                            logging.info(
                                'Global step: {}, epoch step: {}, interval loss: {:.4f}'.format(
                                    global_step, epoch_step, loss_scalar
                                ))

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        max_length = max(inputs['lengths'])
                        oom_time += 1
                        logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                            oom_time, cfg.batch_size, max_length))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        logging.info(str(exception))
                        raise exception
            logging.info('Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(
                (time.time() - btm) / 60, tr_loss))

            # validation with loss
            valid_loss = 0
            valid_data_iterator = self.reader.get_nontranspose_data_iterator(valid_all_batches)

            for batch_idx, dial_batch in enumerate(valid_data_iterator):
                self.model.eval()

                inputs = self.reader.convert_batch_session(dial_batch)
                inputs = self.add_torch_input(inputs)
                outputs = self.model(inputs['contexts_tensor'])
                loss = self.calculate_loss_and_accuracy(outputs, labels=inputs['contexts_tensor'])
                valid_loss += loss.item()

            valid_losses.append(valid_loss)

            print('VALID LOSS: {:.4f}'.format(valid_loss))
            if valid_loss < valid_best_loss:
                valid_best_loss = valid_loss
                print('************ BEST LOSS: {} / EPOCH: {} ************'.format(valid_best_loss, epoch + 1))

                if prev_path != '':
                    self.remove_model(prev_path)
                prev_path = self.save_model(epoch, tr_loss / epoch_step)

    def save_model(self, epoch, loss):
        save_path = os.path.join(
            cfg.exp_path, 'epoch{}_trloss{:.2f}_gpt2'.format(epoch + 1, loss))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        logging.info('Saving model checkpoint to %s', save_path)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        return save_path

    def remove_model(self, path):
        if os.path.exists(path):
            # os.rmdir(path)
            shutil.rmtree(path)
        logging.info('remove model checkpoint of %s', path)

    def greedy_action_sampling_v0(self, inputs, turn_domain):
        context_length = len(inputs['context'])

        # Belief state prediction
        bs_output = self.model.generate(input_ids=inputs['context_tensor'],
                                        max_length=context_length + 60, temperature=0.7,  # top_p=0.9, num_beams=4,
                                        # do_sample=True,
                                        # num_beams=10,
                                        # num_return_sequences=5,
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        eos_token_id=self.tokenizer.encode(['<eos_b>'])[0])

        generated_bs = bs_output[0].cpu().numpy().tolist()
        bspn_gen = self.decode_generated_bspn(generated_bs[context_length - 1:])

        # DB search
        db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(bspn_gen), turn_domain)
        db = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize('<sos_db> ' + db_result + ' <eos_db>')) + self.tokenizer.encode(
            ['<sos_a>'])
        inputs['context_tensor_db'] = torch.tensor([inputs['context'][:-1] + bspn_gen + db]).to(
            self.device)
        context_length = len(inputs['context_tensor_db'][0])

        # Dialogue act / system response generation
        sys_output = self.model.generate(input_ids=inputs['context_tensor_db'],
                                         max_length=context_length + 80, temperature=0.7,
                                         # do_sample=True,
                                         pad_token_id=self.tokenizer.eos_token_id,
                                         eos_token_id=self.tokenizer.encode(['<eos_r>'])[0])

        hidden = self.model(sys_output, get_hidden=True)
        values = self.critic(hidden)
        q_value = values[0, -1, 0].item()

        greedy_output = sys_output[0].cpu().numpy().tolist()
        generated_ar = greedy_output[context_length - 1:]

        return bspn_gen, db, generated_ar, q_value

    def greedy_action_sampling(self, inputs, turn, turn_domain):
        bs = turn['bspn']
        db = turn['db']

        inputs['context_tensor_db'] = torch.tensor(
            [inputs['context'][:-1] + bs + db + self.tokenizer.encode(['<sos_a>'])]).to(self.device)
        context_db_length = len(inputs['context_tensor_db'][0])

        # Dialogue act / system response generation
        sys_output = self.model.generate(input_ids=inputs['context_tensor_db'],
                                         max_length=context_db_length + 80, temperature=0.7,
                                         pad_token_id=self.tokenizer.eos_token_id,
                                         eos_token_id=self.tokenizer.encode(['<eos_r>'])[0])

        hidden = self.model(sys_output, get_hidden=True)
        values = self.critic(hidden)
        q_value = values[0, -1, 0].item()

        greedy_output = sys_output[0].cpu().numpy().tolist()
        generated_ar = greedy_output[context_db_length - 1:]

        return bs, db, generated_ar, q_value

    def q_best_action_sampling_v0(self, inputs, turn, turn_domain, sampling_num=5):
        best_bspn_gen, best_db, best_ar_gen = None, None, None
        q_best = -10e6
        q_values = []
        context_length = len(inputs['context'])

        # Belief state prediction
        bs_output = self.model.generate(input_ids=inputs['context_tensor'],
                                        max_length=context_length + 60, temperature=0.7,
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        eos_token_id=self.tokenizer.encode(['<eos_b>'])[0])

        generated_bs = bs_output[0].cpu().numpy().tolist()
        bspn_gen = self.decode_generated_bspn(generated_bs[context_length - 1:])

        # DB search
        db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(bspn_gen), turn_domain)
        db = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize('<sos_db> ' + db_result + ' <eos_db>')) + self.tokenizer.encode(
            ['<sos_a>'])
        inputs['context_tensor_db'] = torch.tensor([inputs['context'][:-1] + bspn_gen + db]).to(
            self.device)
        context_db_length = len(inputs['context_tensor_db'][0])

        for i in range(sampling_num):
            aspn_output = self.model.generate(input_ids=inputs['context_tensor_db'],
                                              max_length=context_db_length + 40, temperature=0.7,
                                              do_sample=True,
                                              pad_token_id=self.tokenizer.eos_token_id,
                                              eos_token_id=self.tokenizer.encode(['<eos_a>'])[0])

            aspn_gen = aspn_output[0].cpu().numpy().tolist()

            inputs['context_tensor_aspn'] = torch.tensor([aspn_gen + self.tokenizer.encode(['<sos_r>'])]).to(
                self.device)
            context_aspn_length = len(aspn_gen)

            sys_output = self.model.generate(input_ids=inputs['context_tensor_aspn'],
                                             max_length=context_aspn_length + 50, temperature=0.7,
                                             pad_token_id=self.tokenizer.eos_token_id,
                                             eos_token_id=self.tokenizer.encode(['<eos_r>'])[0])

            ar_gen = sys_output[0].cpu().numpy().tolist()
            ar_gen = ar_gen[context_db_length - 1:]

            if self.tokenizer.encode(['<eos_r>'])[0] != ar_gen[-1]:
                # print ('#####')
                # print (self.tokenizer.encode(['<eos_r>'])[0])
                # print (ar_gen[-1])
                # print (ar_gen)
                # print (self.tokenizer.decode(ar_gen))
                q_value = -10e5
            else:
                # Evaluate Q-value of actions
                hidden = self.model(sys_output, get_hidden=True)
                values = self.critic(hidden)
                q_value = values[0, -1, 0].item()

            # if not self.tokenizer.encode(['<eos_r>'])[0] != ar_gen[-1]:
            #     ar_gen.append(self.tokenizer.encode(['<eos_r>'])[0])

            q_values.append(q_value)

            # Check generated samples and q-values
            # print ('######### SAMPLE {} ########'.format(i))
            # print (self.tokenizer.decode(inputs['context']))
            # print (self.tokenizer.decode(bspn_gen))
            # print (self.tokenizer.decode(sys_output[0].cpu().numpy().tolist()[context_db_length - 1:]))
            # print (q_value)
            # print ()

            # Sampled best action by Q-value
            if q_value > q_best:
                best_bspn_gen = copy.deepcopy(bspn_gen)
                best_db = copy.deepcopy(db)
                best_ar_gen = copy.deepcopy(ar_gen)
                q_best = q_value

        return best_bspn_gen, best_db, best_ar_gen, q_best, q_values

    def q_best_action_sampling(self, inputs, turn, turn_domain, sampling_num=5):
        _, _, best_ar_gen, q_best = self.greedy_action_sampling(inputs, turn, turn_domain)

        q_values = []

        bs = turn['bspn']
        db = turn['db']

        inputs['context_tensor_db'] = torch.tensor(
            [inputs['context'][:-1] + bs + db + self.tokenizer.encode(['<sos_a>'])]).to(self.device)
        context_db_length = len(inputs['context_tensor_db'][0])

        for i in range(sampling_num - 1):
            aspn_output = self.model.generate(input_ids=inputs['context_tensor_db'],
                                              max_length=context_db_length + 40, temperature=0.7,
                                              do_sample=True,
                                              pad_token_id=self.tokenizer.eos_token_id,
                                              eos_token_id=self.tokenizer.encode(['<eos_a>'])[0])

            aspn_gen = aspn_output[0].cpu().numpy().tolist()

            inputs['context_tensor_aspn'] = torch.tensor([aspn_gen + self.tokenizer.encode(['<sos_r>'])]).to(
                self.device)
            context_aspn_length = len(aspn_gen)

            sys_output = self.model.generate(input_ids=inputs['context_tensor_aspn'],
                                             max_length=context_aspn_length + 50, temperature=0.7,
                                             pad_token_id=self.tokenizer.eos_token_id,
                                             eos_token_id=self.tokenizer.encode(['<eos_r>'])[0])

            ar_gen = sys_output[0].cpu().numpy().tolist()
            ar_gen = ar_gen[context_db_length - 1:]

            if self.tokenizer.encode(['<eos_r>'])[0] != ar_gen[-1]:
                # print ('#####')
                # print (self.tokenizer.encode(['<eos_r>'])[0])
                # print (ar_gen[-1])
                # print (ar_gen)
                # print (self.tokenizer.decode(ar_gen))
                q_value = -10e5
            else:
                # Evaluate Q-value of actions
                hidden = self.model(sys_output, get_hidden=True)
                values = self.critic(hidden)
                q_value = values[0, -1, 0].item()

            # if not self.tokenizer.encode(['<eos_r>'])[0] != ar_gen[-1]:
            #     ar_gen.append(self.tokenizer.encode(['<eos_r>'])[0])

            q_values.append(q_value)

            # Check generated samples and q-values
            # print ('######### SAMPLE {} ########'.format(i))
            # print (self.tokenizer.decode(inputs['context']))
            # print (self.tokenizer.decode(bspn_gen))
            # print (self.tokenizer.decode(sys_output[0].cpu().numpy().tolist()[context_db_length - 1:]))
            # print (q_value)
            # print ()

            # Sampled best action by Q-value
            if q_value > q_best:
                # best_bspn_gen = copy.deepcopy(bspn_gen)
                # best_db = copy.deepcopy(db)
                best_ar_gen = copy.deepcopy(ar_gen)
                q_best = q_value

        return bs, db, best_ar_gen, q_best, q_values

    def self_generate_v0(self, iteration=0, seed=0, sampling_num=5):
        self.model.eval()
        self.critic = torch.load('policy_evaluation_weight/critic_iter_{}_seed_{}'.format(iteration, seed),
                                 map_location=self.device)
        self.critic.eval()

        if iteration == 0:
            with open('data/self-generation/train_iter_{}.pickle'.format(iteration), 'rb') as f:
                train_data = pickle.load(f)
        else:
            with open('data/self-generation/train_iter_{}_seed_{}.pickle'.format(iteration, seed), 'rb') as f:
                train_data = pickle.load(f)

        scores = np.load('data/scores/cum_scores_iter_{}_seed_{}.npy'.format(iteration, seed),
                         allow_pickle=True).tolist()

        original_data = train_data
        original_eval_data = copy.deepcopy(original_data)
        copy_data = copy.deepcopy(original_data)
        updated_data = copy.deepcopy(original_data)

        total_number = 0
        fail_num = 0
        better_number = 0
        negative_num = 0

        with torch.no_grad():
            for dial_idx, dialog in enumerate(copy_data):
                pv_turn = {}
                result_collection = {}
                original_result_collection = {}
                skip_update = False

                if (dial_idx) % 500 == 0:
                    print('#### COPMLETE GENERATION DIALOG {} ####'.format(dial_idx))

                # _, _, original_cum_scores = self.evaluator.training_data_eval(original_eval_data[dial_idx:dial_idx + 1])
                # original_eval = original_cum_scores[0]
                original_eval = scores[dial_idx]

                if original_eval[-1] == 1.0:
                    continue

                fail_num += 1

                for turn_idx, turn in enumerate(dialog):
                    first_turn = (turn_idx == 0)

                    inputs = self.reader.convert_turn_eval(turn, pv_turn, first_turn)
                    inputs = self.add_torch_input_eval(inputs)

                    # TODO: Select best-Q action including original data
                    original_inputs = {}
                    original_inputs['context'] = copy.deepcopy(inputs['context'])
                    original_inputs['context'] = original_inputs['context'][:-1] + turn['bspn'] + turn['db'] + turn[
                        'aspn'] + turn['resp']

                    original_inputs = self.add_torch_input_eval(original_inputs)

                    hidden = self.model(original_inputs['context_tensor'], get_hidden=True)
                    values = self.critic(hidden)
                    original_q_value = values[0, -1, 0].item()

                    # Sampling action with Q-values
                    bspn_gen, db, generated_ar, q_best, q_values = self.q_best_action_sampling(inputs, turn,
                                                                                               turn['turn_domain'],
                                                                                               sampling_num=sampling_num)

                    try:
                        decoded = self.decode_generated_act_resp(generated_ar)
                        decoded['bspn'] = bspn_gen
                    except ValueError as exception:
                        logging.info(str(exception))
                        logging.info(self.tokenizer.decode(generated_ar))
                        decoded = {'resp': [], 'bspn': [], 'aspn': []}

                    if original_q_value > q_best:
                        decoded = {}
                        decoded['bspn'] = turn['bspn']
                        decoded['aspn'] = turn['aspn']
                        decoded['resp'] = turn['resp']
                        db = turn['db']
                    # else:
                    #     print ("CHANGE!!")
                    # else:
                    #     turn['pointer'] = self.reader.bspan_to_PointerVector(
                    #         self.tokenizer.decode(decoded['bspn'][1:-1]),
                    #         turn['turn_domain'])

                    turn['resp_gen'] = decoded['resp']
                    turn['bspn_gen'] = turn['bspn'] if cfg.use_true_curr_bspn else decoded['bspn']
                    turn['aspn_gen'] = turn['aspn'] if cfg.use_true_curr_aspn else decoded['aspn']
                    turn['dspn_gen'] = turn['dspn']

                    turn['pointer'] = self.reader.bspan_to_PointerVector(self.tokenizer.decode(turn['bspn_gen'][1:-1]),
                                                                         turn['turn_domain'])
                    # value = self.reader.db.pointerBack(turn['pointer'], turn['turn_domain'][-1])

                    turn['bsdx'] = turn['bspn_gen']

                    pv_turn['labels'] = inputs['labels']  # all true previous context
                    pv_turn['resp'] = turn['resp'] if cfg.use_true_prev_resp else decoded['resp']
                    pv_turn['bspn'] = turn['bspn'] if cfg.use_true_prev_bspn else decoded['bspn']
                    pv_turn['db'] = turn['db'] if cfg.use_true_curr_bspn else db
                    pv_turn['aspn'] = turn['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']

                    pv_turn['pointer'] = turn['pointer']
                    pv_turn['bsdx'] = turn['bsdx']

                    if not 50307 in decoded['resp']:  # 50307: '<eos_r>'
                        skip_update = True

                result_collection.update(self.reader.inverse_transpose_turn(dialog))
                results, _ = self.reader.wrap_result_lm(result_collection)
                _, _, updated_cum_scores = self.evaluator.temporal_eval(results, cum_score=True)
                argmax_q_eval = updated_cum_scores[-1]

                if original_eval[-1] < argmax_q_eval and not skip_update:
                    better_number += 1
                    print('UP', original_eval[-1], argmax_q_eval)
                    for idx in range(len(updated_data[dial_idx])):
                        updated_data[dial_idx][idx]['resp'] = [50313] + self.tokenizer.encode(
                            results[idx + 1]['resp_gen']) + [50307]  # 50313 / 50307
                        updated_data[dial_idx][idx]['bspn'] = [50314] + self.tokenizer.encode(
                            results[idx + 1]['bspn_gen']) + [50308]  # 50314 / 50308
                        updated_data[dial_idx][idx]['bsdx'] = [50314] + self.tokenizer.encode(
                            results[idx + 1]['bspn_gen']) + [50308]  # None
                        updated_data[dial_idx][idx]['aspn'] = [50315] + self.tokenizer.encode(
                            results[idx + 1]['aspn_gen']) + [50309]  # 50315 / 50309
                        updated_data[dial_idx][idx]['dspn'] = [50316] + self.tokenizer.encode(
                            results[idx + 1]['dspn_gen']) + [50311]  # 50316 / 50311
                        updated_data[dial_idx][idx]['pointer'] = results[idx + 1]['pointer']
                        turn_domain = result_collection[list(result_collection.keys())[0]][idx]['turn_domain']
                        updated_data[dial_idx][idx]['turn_domain'] = turn_domain
                        updated_data[dial_idx][idx]['db'] = results[idx + 1]['db']  # 50317 / 50318
                elif original_eval[-1] > argmax_q_eval:
                    negative_num += 1
                    # print('DOWN', original_eval[-1], argmax_q_eval)

                print('CURRENT DIALOG INDEX: {} / {} UPDATED COUNT (U / N / T): ({} / {} / {})'.format(dial_idx,
                                                                                                       len(original_data),
                                                                                                       better_number,
                                                                                                       negative_num,
                                                                                                       fail_num))
        print('#### COPMLETE TO REVISE THE UNSUCCESSFUL DIALOGUES {} ####'.format(len(original_data)))

        with open('data/self-generation/train_iter_{}_seed_{}.pickle'.format(iteration + 1, seed), 'wb') as f:
            pickle.dump(updated_data, f, pickle.HIGHEST_PROTOCOL)

    def self_generate(self, iteration=0, seed=0, sampling_num=5):
        self.model.eval()
        self.critic = torch.load('policy_evaluation_weight/critic_iter_{}_seed_{}'.format(iteration, seed),
                                 map_location=self.device)
        self.critic.eval()

        if iteration == 0:
            with open('data/self-generation/train_iter_{}.pickle'.format(iteration), 'rb') as f:
                train_data = pickle.load(f)
        else:
            with open('data/self-generation/train_iter_{}_seed_{}.pickle'.format(iteration, seed), 'rb') as f:
                train_data = pickle.load(f)

        scores = np.load('data/scores/cum_scores_iter_{}_seed_{}.npy'.format(iteration, seed),
                         allow_pickle=True).tolist()

        original_data = train_data
        original_eval_data = copy.deepcopy(original_data)
        copy_data = copy.deepcopy(original_data)
        updated_data = copy.deepcopy(original_data)

        total_number = 0
        fail_num = 0
        better_number = 0
        negative_num = 0

        with torch.no_grad():
            for dial_idx, dialog in enumerate(copy_data):
                pv_turn = {}
                result_collection = {}
                original_result_collection = {}
                skip_update = False

                if (dial_idx) % 500 == 0:
                    print('#### COPMLETE GENERATION DIALOG {} ####'.format(dial_idx))

                # _, _, original_cum_scores = self.evaluator.training_data_eval(original_eval_data[dial_idx:dial_idx + 1])
                # original_eval = original_cum_scores[0]
                original_eval = scores[dial_idx]

                if original_eval[-1] == 1.0:
                    continue

                fail_num += 1

                for turn_idx, turn in enumerate(dialog):
                    first_turn = (turn_idx == 0)

                    inputs = self.reader.convert_turn_eval(turn, pv_turn, first_turn)
                    inputs = self.add_torch_input_eval(inputs)

                    # TODO: Select best-Q action including original data
                    original_inputs = {}
                    original_inputs['context'] = copy.deepcopy(inputs['context'])
                    original_inputs['context'] = original_inputs['context'][:-1] + turn['bspn'] + turn['db'] + turn[
                        'aspn'] + turn['resp']

                    original_inputs = self.add_torch_input_eval(original_inputs)

                    hidden = self.model(original_inputs['context_tensor'], get_hidden=True)
                    values = self.critic(hidden)
                    original_q_value = values[0, -1, 0].item()

                    # Sampling action with Q-values
                    bspn_gen, db, generated_ar, q_best, q_values = self.q_best_action_sampling(inputs, turn,
                                                                                               turn['turn_domain'],
                                                                                               sampling_num=sampling_num)

                    try:
                        decoded = self.decode_generated_act_resp(generated_ar)
                        decoded['bspn'] = bspn_gen
                    except ValueError as exception:
                        logging.info(str(exception))
                        logging.info(self.tokenizer.decode(generated_ar))
                        decoded = {'resp': [], 'bspn': [], 'aspn': []}

                    if original_q_value > q_best:
                        decoded = {}
                        decoded['bspn'] = turn['bspn']
                        decoded['aspn'] = turn['aspn']
                        decoded['resp'] = turn['resp']
                        db = turn['db']
                    # else:
                    #     print ("CHANGE!!")
                    # else:
                    #     turn['pointer'] = self.reader.bspan_to_PointerVector(
                    #         self.tokenizer.decode(decoded['bspn'][1:-1]),
                    #         turn['turn_domain'])

                    turn['resp_gen'] = decoded['resp']
                    turn['bspn_gen'] = turn['bspn'] if cfg.use_true_curr_bspn else decoded['bspn']
                    turn['aspn_gen'] = turn['aspn'] if cfg.use_true_curr_aspn else decoded['aspn']
                    turn['dspn_gen'] = turn['dspn']

                    turn['pointer'] = self.reader.bspan_to_PointerVector(self.tokenizer.decode(turn['bspn_gen'][1:-1]),
                                                                         turn['turn_domain'])
                    # value = self.reader.db.pointerBack(turn['pointer'], turn['turn_domain'][-1])

                    turn['bsdx'] = turn['bspn_gen']

                    pv_turn['labels'] = inputs['labels']  # all true previous context
                    pv_turn['resp'] = turn['resp'] if cfg.use_true_prev_resp else decoded['resp']
                    pv_turn['bspn'] = turn['bspn'] if cfg.use_true_prev_bspn else decoded['bspn']
                    pv_turn['db'] = turn['db'] if cfg.use_true_curr_bspn else db
                    pv_turn['aspn'] = turn['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']

                    pv_turn['pointer'] = turn['pointer']
                    pv_turn['bsdx'] = turn['bsdx']

                    if not 50307 in decoded['resp']:  # 50307: '<eos_r>'
                        skip_update = True

                result_collection.update(self.reader.inverse_transpose_turn(dialog))
                results, _ = self.reader.wrap_result_lm(result_collection)
                _, _, updated_cum_scores = self.evaluator.temporal_eval(results, cum_score=True)
                argmax_q_eval = updated_cum_scores[-1]

                if original_eval[-1] < argmax_q_eval and not skip_update:
                    better_number += 1
                    print('UP', original_eval[-1], argmax_q_eval)
                    for idx in range(len(updated_data[dial_idx])):
                        updated_data[dial_idx][idx]['resp'] = [50313] + self.tokenizer.encode(
                            results[idx + 1]['resp_gen']) + [50307]  # 50313 / 50307
                        updated_data[dial_idx][idx]['bspn'] = [50314] + self.tokenizer.encode(
                            results[idx + 1]['bspn_gen']) + [50308]  # 50314 / 50308
                        updated_data[dial_idx][idx]['bsdx'] = [50314] + self.tokenizer.encode(
                            results[idx + 1]['bspn_gen']) + [50308]  # None
                        updated_data[dial_idx][idx]['aspn'] = [50315] + self.tokenizer.encode(
                            results[idx + 1]['aspn_gen']) + [50309]  # 50315 / 50309
                        updated_data[dial_idx][idx]['dspn'] = [50316] + self.tokenizer.encode(
                            results[idx + 1]['dspn_gen']) + [50311]  # 50316 / 50311
                        updated_data[dial_idx][idx]['pointer'] = results[idx + 1]['pointer']
                        turn_domain = result_collection[list(result_collection.keys())[0]][idx]['turn_domain']
                        updated_data[dial_idx][idx]['turn_domain'] = turn_domain
                        updated_data[dial_idx][idx]['db'] = results[idx + 1]['db']  # 50317 / 50318
                elif original_eval[-1] > argmax_q_eval:
                    negative_num += 1
                    # print('DOWN', original_eval[-1], argmax_q_eval)

                print('CURRENT DIALOG INDEX: {} / {} UPDATED COUNT (U / N / T): ({} / {} / {})'.format(dial_idx,
                                                                                                       len(original_data),
                                                                                                       better_number,
                                                                                                       negative_num,
                                                                                                       fail_num))
        print('#### COPMLETE TO REVISE THE UNSUCCESSFUL DIALOGUES {} ####'.format(len(original_data)))

        with open('data/self-generation/train_iter_{}_seed_{}.pickle'.format(iteration + 1, seed), 'wb') as f:
            pickle.dump(updated_data, f, pickle.HIGHEST_PROTOCOL)

    def policy_evaluation(self, iteration=0, seed=0):
        """
        Policy evaluation with revised dataset
        """
        scores = np.load('data/scores/cum_scores_iter_{}_seed_{}.npy'.format(iteration, seed),
                         allow_pickle=True).tolist()

        if iteration == 0:
            with open('data/self-generation/train_iter_{}.pickle'.format(iteration), 'rb') as f:
                self.reader.train = pickle.load(f)
        else:
            with open('data/self-generation/train_iter_{}_seed_{}.pickle'.format(iteration, seed), 'rb') as f:
                self.reader.train = pickle.load(f)

        all_batches = self.reader.get_transition_batches('train', scores, state_until_db=True)

        optimizer, scheduler, critic_optimizer = self.get_optimizers()
        self.target_critic.eval()

        logging.info("***** Policy Evaluation *****")
        logging.info("  Num Epochs = %d", cfg.epoch_num)

        for epoch in range(10):
            epoch_loss = 0.0
            skipped_count = 0
            self.critic.zero_grad()

            data_iterator = self.reader.get_nontranspose_data_iterator(all_batches)

            for batch_idx, dial_batch in tqdm(enumerate(data_iterator), total=len(all_batches)):
                inputs = self.reader.convert_policy_eval_batch_session(dial_batch)
                skip_condition = False
                if inputs['sa_np'].shape[-1] > 1024 or np.max(inputs['eos_sa']) >= 1024 \
                        or inputs['next_sa_np'].shape[-1] > 1024 or np.max(inputs['eos_next_sa']) >= 1024:
                    skipped_count += 1
                    skip_condition = True

                # for i in range(len(inputs['sampled_next_sa_np_list'])):
                #     for idx, next_sa_idx in enumerate(inputs['sampled_next_sa_length_list'][i]):
                #         if next_sa_idx - 1 > 1023:
                #             skip_condition = True

                if skip_condition:
                    continue

                self.critic.train()

                inputs = self.add_torch_transition_input(inputs)

                hidden = self.model(inputs['sa_tensor'], get_hidden=True)
                values = self.critic(hidden)

                next_hidden = self.model(inputs['next_sa_tensor'], get_hidden=True)
                next_values = self.target_critic(next_hidden)
                masked_target = torch.sum(inputs['reward_tensor'], 1) \
                                + self.gamma * torch.sum(inputs['terminal_mask_tensor']
                                                         * next_values * inputs['mask_next_sa_tensor'], 1)
                masked_value = torch.sum(inputs['mask_sa_tensor'] * values, 1)

                loss = torch.sum((masked_value - masked_target) ** 2)
                loss.backward()
                epoch_loss += loss.item()

                critic_optimizer.step()
                scheduler.step()
                critic_optimizer.zero_grad()

                self.soft_update_target()

            print('Training epoch: {} Loss: {:.4f} SKIP COUNT: {}'.format(epoch, epoch_loss / (batch_idx + 1),
                                                                          skipped_count))

            torch.save(self.critic, 'policy_evaluation_weight/critic_iter_{}_seed_{}'.format(iteration, seed))

    def generate_rewards(self, iteration=0, seed=0):
        if iteration == 0:
            with open('data/self-generation/train_iter_{}.pickle'.format(iteration), 'rb') as f:
                data = pickle.load(f)
        else:
            with open('data/self-generation/train_iter_{}_seed_{}.pickle'.format(iteration, seed), 'rb') as f:
                data = pickle.load(f)

        successes, matches, scores = self.evaluator.training_data_metric(data, self.tokenizer)

        np.save('data/scores/cum_successes_iter_{}_seed_{}.npy'.format(iteration, seed), np.array(successes))
        np.save('data/scores/cum_matches_iter_{}_seed_{}.npy'.format(iteration, seed), np.array(matches))
        np.save('data/scores/cum_scores_iter_{}_seed_{}.npy'.format(iteration, seed), np.array(scores))

        print('Success: {:.4f} / Match: {:.4f}'.format(np.mean(successes), np.mean(matches)))

    def validate(self, data='dev', do_test=False):
        # predict one dialog/ one turn at a time
        self.model.eval()
        eval_data = self.reader.get_eval_data(data)

        set_stats = self.reader.set_stats[data]
        logging.info("***** Running Evaluation *****")
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        # logging.info("  Num Dialogs = %d", set_stats['num_dials'])

        # valid_losses = []
        btm = time.time()
        result_collection = {}
        with torch.no_grad():
            for dial_idx, dialog in enumerate(tqdm(eval_data)):
                pv_turn = {}
                for turn_idx, turn in enumerate(dialog):
                    first_turn = (turn_idx == 0)
                    inputs = self.reader.convert_turn_eval(
                        turn, pv_turn, first_turn)
                    inputs = self.add_torch_input_eval(inputs)
                    # fail to generate new tokens, if max_length not set
                    context_length = len(inputs['context'])
                    if cfg.use_true_curr_bspn:  # generate act, response
                        max_len = 60
                        if not cfg.use_true_curr_aspn:
                            max_len = 80

                        outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                                      max_length=context_length + max_len, temperature=0.7,
                                                      # top_p=0.9, num_beams=4,
                                                      pad_token_id=self.tokenizer.eos_token_id,
                                                      eos_token_id=self.tokenizer.encode(['<eos_r>'])[0])

                        # resp_gen, need to trim previous context
                        generated = outputs[0].cpu().numpy().tolist()
                        generated = generated[context_length - 1:]

                        try:
                            decoded = self.decode_generated_act_resp(generated)
                        except ValueError as exception:
                            logging.info(str(exception))
                            logging.info(self.tokenizer.decode(generated))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}

                    else:  # predict bspn, access db, then generate act and resp
                        outputs = self.model.generate(input_ids=inputs['context_tensor'],
                                                      max_length=context_length + 60, temperature=0.7,
                                                      # top_p=0.9, num_beams=4,
                                                      # do_sample=True,
                                                      pad_token_id=self.tokenizer.eos_token_id,
                                                      eos_token_id=self.tokenizer.encode(['<eos_b>'])[0])
                        generated_bs = outputs[0].cpu().numpy().tolist()
                        # generated_bs = generated_bs[context_length-1:]
                        bspn_gen = self.decode_generated_bspn(generated_bs[context_length - 1:])
                        # check DB result
                        if cfg.use_true_db_pointer:
                            db = turn['db']
                        else:
                            db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(bspn_gen),
                                                                       turn['turn_domain'])
                            db = self.tokenizer.convert_tokens_to_ids(
                                self.tokenizer.tokenize('<sos_db> ' + db_result + ' <eos_db>')) + self.tokenizer.encode(
                                ['<sos_a>'])
                        inputs['context_tensor_db'] = torch.tensor([inputs['context'][:-1] + bspn_gen + db]).to(
                            self.device)
                        context_length = len(inputs['context_tensor_db'][0])
                        outputs_db = self.model.generate(input_ids=inputs['context_tensor_db'],
                                                         max_length=context_length + 80, temperature=0.7,
                                                         # top_p=0.9, num_beams=5,
                                                         # do_sample=True,
                                                         pad_token_id=self.tokenizer.eos_token_id,
                                                         eos_token_id=self.tokenizer.encode(['<eos_r>'])[0])
                        generated_ar = outputs_db[0].cpu().numpy().tolist()
                        generated_ar = generated_ar[context_length - 1:]
                        if self.tokenizer.encode(['<eos_r>'])[0] != generated_ar[-1]:
                            generated_ar.append(self.tokenizer.encode(['<eos_r>'])[0])
                        # print (self.tokenizer.decode(generated_ar))
                        try:
                            decoded = self.decode_generated_act_resp(generated_ar)
                            decoded['bspn'] = bspn_gen
                        except ValueError as exception:
                            logging.info(str(exception))
                            logging.info(self.tokenizer.decode(generated_ar))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}

                    turn['resp_gen'] = decoded['resp']
                    turn['bspn_gen'] = turn['bspn'] if cfg.use_true_curr_bspn else decoded['bspn']
                    turn['aspn_gen'] = turn['aspn'] if cfg.use_true_curr_aspn else decoded['aspn']
                    turn['dspn_gen'] = turn['dspn']

                    # check DB results
                    # db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                    # if db_result[0] == 1: # no match
                    #     print('gt:', self.tokenizer.decode(turn['aspn']), '     |gen:', self.tokenizer.decode(decoded['aspn']))
                    #     print('gen_resp: ', self.tokenizer.decode(decoded['resp']))
                    #     print('gt_resp: ', self.tokenizer.decode(turn['resp']), '\n')

                    pv_turn['labels'] = inputs['labels']  # all true previous context
                    pv_turn['resp'] = turn['resp'] if cfg.use_true_prev_resp else decoded['resp']
                    pv_turn['bspn'] = turn['bspn'] if cfg.use_true_prev_bspn else decoded['bspn']
                    pv_turn['db'] = turn['db'] if cfg.use_true_prev_bspn else db
                    pv_turn['aspn'] = turn['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']

                result_collection.update(
                    self.reader.inverse_transpose_turn(dialog))

        logging.info("inference time: {:.2f} min".format((time.time() - btm) / 60))
        # score
        btm = time.time()
        results, _ = self.reader.wrap_result_lm(result_collection)

        metric_result = self.evaluator._get_metric_results(results)

        bleu, success, match = self.evaluator.validation_metric(results)
        logging.info("Scoring time: {:.2f} min".format((time.time() - btm) / 60))
        score = 0.5 * (success + match) + bleu
        valid_loss = 130 - score
        logging.info('validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f' % (
            match, success, bleu, score))
        eval_results = {}
        eval_results['bleu'] = bleu
        eval_results['success'] = success
        eval_results['match'] = match
        eval_results['score'] = score
        eval_results['result'] = 'validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f' % (
        match, success, bleu, score)

        model_setting, epoch_setting = cfg.eval_load_path.split('/')[1], cfg.eval_load_path.split('/')[2]
        eval_on = '-'.join(cfg.exp_domains)
        if data == 'test':
            eval_on += '_test'
        if not os.path.exists(cfg.log_path):
            os.mkdir(cfg.log_path)
        log_file_name = os.path.join(cfg.log_path, model_setting + '-' + eval_on + '.json')
        if os.path.exists(log_file_name):
            eval_to_json = json.load(open(log_file_name, 'r'))
            eval_to_json[epoch_setting] = eval_results
            json.dump(eval_to_json, open(log_file_name, 'w'), indent=2)
        else:
            eval_to_json = {}
            eval_to_json[epoch_setting] = eval_results
            json.dump(eval_to_json, open(log_file_name, 'w'), indent=2)
        logging.info('update eval results to {}'.format(log_file_name))
        return eval_results

    def decode_generated_act_resp(self, generated):
        """
        decode generated
        return decoded['resp'] ('bspn', 'aspn')
        """
        decoded = {}
        eos_a_id = self.tokenizer.encode(['<eos_a>'])[0]
        eos_r_id = self.tokenizer.encode(['<eos_r>'])[0]
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]

        # eos_r may not exists if gpt2 generated repetitive words.
        if eos_r_id in generated:
            eos_r_idx = generated.index(eos_r_id)
        else:
            eos_r_idx = len(generated) - 1
            logging.info('eos_r not in generated: ' + self.tokenizer.decode(generated))
        # eos_r_idx = generated.index(eos_r_id) if eos_r_id in generated else len(generated)-1

        if cfg.use_true_curr_aspn:  # only predict resp
            decoded['resp'] = generated[: eos_r_idx + 1]
        else:  # predicted aspn, resp
            eos_a_idx = generated.index(eos_a_id)
            decoded['aspn'] = generated[: eos_a_idx + 1]
            decoded['resp'] = generated[eos_a_idx + 1: eos_r_idx + 1]
        # if cfg.use_true_curr_bspn:

        # else:  # predict bspn aspn resp
        #     eos_b_idx = generated.index(eos_b_id)
        #     eos_a_idx = generated.index(eos_a_id)
        #     decoded['bspn'] = generated[: eos_b_idx+1]
        #     decoded['aspn'] = generated[eos_b_idx+1: eos_a_idx+1]
        #     decoded['resp'] = generated[eos_a_idx+1: eos_r_idx+1]
        return decoded

    def decode_generated_bspn(self, generated):
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]
        if eos_b_id in generated:
            eos_b_idx = generated.index(eos_b_id)
        else:
            eos_b_idx = len(generated) - 1
        return generated[: eos_b_idx + 1]


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
            cfg.exp_path = os.path.join(experiments_path,
                                        '{}_iter_{}_seed_{}'.format(cfg.exp_no, cfg.iteration, cfg.seed))
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
    m = UBAR(device)

    if args.mode == 'train':  # train
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
        logging.info(
            "Generate setting: \n\t use true_prev_bspn={} \n\t use true_prev_aspn={} \n\t use true_db_pointer={} \n\t use true_prev_resp={} \n\t use true_curr_bspn={} \n\t use true_curr_aspn={} \n\t use_all_previous_context={}".format(
                cfg.use_true_prev_bspn, cfg.use_true_prev_aspn, cfg.use_true_db_pointer, cfg.use_true_prev_resp,
                cfg.use_true_curr_bspn, cfg.use_true_curr_aspn, cfg.use_all_previous_context
            ))

        if cfg.context_scheme == 'UBARU':
            m.validate('test')


if __name__ == "__main__":
    main()