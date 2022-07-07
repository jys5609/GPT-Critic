import copy
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from eval import MultiWozEvaluator
from reader import MultiWozReader

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import shutil
from model_utils import add_torch_input
from model_utils import add_torch_input_with_advantage
from model_utils import add_torch_input_eval
from model_utils import add_torch_transition_input

import os
import time
import logging
import json
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import global_config as cfg

import warnings

warnings.filterwarnings("ignore")


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256, bias=True)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(256, 1, bias=True)

    def forward(self, input):
        return self.fc2(self.fc1(input))


class GPT2Policy(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=True,
        get_hidden=False):
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs[0]

        if get_hidden:
            return hidden_states

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


class WeightedBC(object):
    def __init__(self, device):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path)
        self.reader = MultiWozReader(self.tokenizer)
        self.policy = GPT2Policy.from_pretrained(cfg.gpt_path)
        self.gamma = 0.99

        self.critic = Critic()
        self.target_critic = Critic()

        if cfg.mode == 'train':
            self.policy.resize_token_embeddings(len(self.tokenizer))
        self.policy.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)
        self.soft_update_target(tau=1.0)

        self.evaluator = MultiWozEvaluator(self.reader)
        if cfg.save_log and cfg.mode == 'train':
            self.tb_writer = SummaryWriter(log_dir='./log')
        else:
            self.tb_writer = None

    def get_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.policy.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.policy.named_parameters() if any(nd in n for nd in no_decay)],
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

    def soft_update_target(self, tau=0.1):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def log_first_inputs(self, inputs):
        tokenizer = self.tokenizer
        logging.info("**** Input Examples: ****")
        for context in inputs['contexts'][:4]:
            samples = tokenizer.decode(context)
            logging.info(samples)

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
            return loss

        else:
            lm_logits = outputs[0]

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_advantage_weights = advantage_weights[..., 1:].contiguous()

            pad_id = cfg.pad_id

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss = loss * shift_advantage_weights.view(-1)
            loss = torch.sum(loss)

            not_ignore = shift_labels.ne(pad_id)
            num_targets = not_ignore.long().sum().item()

            loss /= num_targets
            return loss

    def train(self, seed=0, binary_filtering=False):
        with open('data/self-generation/train.pickle', 'rb') as f:
            self.reader.train = pickle.load(f)

        self.policy.eval()
        self.critic = torch.load('policy_evaluation_weight_wbc/critic_seed_{}'.format(seed), map_location=self.device)
        self.critic.eval()

        ### Construct dataset with calculating advantage value for train
        with torch.no_grad():
            for dial_idx, dialog in enumerate(self.reader.train):
                pv_turn = {}

                for turn_idx, turn in enumerate(dialog):
                    first_turn = (turn_idx == 0)

                    inputs = self.reader.convert_turn_eval(turn, pv_turn, first_turn)
                    inputs = add_torch_input_eval(self, inputs)

                    original_inputs = {}
                    original_inputs['context'] = copy.deepcopy(inputs['context'])
                    original_inputs['context'] = original_inputs['context'][:-1] + turn['bspn'] + turn['db'] + turn[
                        'aspn'] + turn['resp']

                    original_inputs = add_torch_input_eval(self, original_inputs)

                    hidden = self.policy(original_inputs['context_tensor'], get_hidden=True)
                    values = self.critic(hidden)
                    q_value = values[0, -1, 0].item()

                    if binary_filtering:
                        advantage = 1 if q_value > 0 else 0
                    else:
                        advantage = np.exp(q_value)
                        # advantage = np.exp(q_value * 5)

                    # turn['advantage'] = [advantage for _ in range(len(inputs['context']))]
                    turn['advantage'] = advantage

                    pv_turn['context'] = inputs['context']
                    pv_turn['labels'] = inputs['labels']
                    pv_turn['resp'] = turn['resp']
                    pv_turn['bspn'] = turn['bspn']
                    pv_turn['db'] = turn['db']
                    pv_turn['aspn'] = turn['aspn']

                    pv_turn['pointer'] = turn['pointer']
                    pv_turn['bsdx'] = turn['bsdx']

        print('COMPLETE ADVANTAGE FOR TRAINSET')

        ### Training with advantage weight of training dataset
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
        logging.info("  Gradient Accumulation steps = %d",
                     cfg.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d",
                     set_stats['num_dials'] * cfg.epoch_num // (cfg.gradient_accumulation_steps * cfg.batch_size))

        # tb writer
        if self.tb_writer is not None:
            self.tb_writer.add_text('cfg', json.dumps(cfg.__dict__, indent=2))
            # self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        log_inputs = 2
        global_step = 0
        valid_best_loss = 1e10
        prev_path = ''
        valid_losses = []

        for epoch in range(cfg.epoch_num):
            if len(valid_losses) > 10 and all(v > valid_best_loss for v in valid_losses[-10:]):
                print('************ EARLY STOPPING ************')
                break

            epoch_step = 0
            tr_loss = 0.0
            logging_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.policy.zero_grad()

            data_iterator = self.reader.get_nontranspose_data_iterator(all_batches)

            for batch_idx, dial_batch in enumerate(data_iterator):
                inputs = self.reader.convert_batch_session_with_advantage(dial_batch)
                try:  # avoid OOM
                    self.policy.train()
                    if log_inputs > 0:  # log inputs for the very first two turns
                        self.log_first_inputs(inputs)
                        log_inputs -= 1

                    inputs = add_torch_input_with_advantage(self, inputs)
                    outputs = self.policy(inputs['contexts_tensor'])
                    loss = self.calculate_loss_and_accuracy(outputs, labels=inputs['contexts_tensor'],
                                                            advantage_weights=inputs['advantage_weights_tensor'])
                    loss.backward()
                    tr_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
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
                self.policy.eval()

                inputs = self.reader.convert_batch_session(dial_batch)
                inputs = add_torch_input(self, inputs)
                outputs = self.policy(inputs['contexts_tensor'])
                loss = self.calculate_loss_and_accuracy(outputs, labels=inputs['contexts_tensor'])
                valid_loss += loss.item()

            valid_losses.append(valid_loss)

            print('VALID LOSS: {:.4f}'.format(valid_loss))
            if valid_loss < valid_best_loss:
                valid_best_loss = valid_loss
                print('************ BEST LOSS: {} / EPOCH: {} ************'.format(valid_best_loss, epoch + 1))

                if prev_path != '':
                    self.remove_model(prev_path)
                prev_path = self.save_model()

    def save_model(self):
        save_path = os.path.join(cfg.exp_path, 'best_model')
        if not os.path.exists(cfg.exp_path):
            os.mkdir(cfg.exp_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        logging.info('Saving model checkpoint to %s', save_path)

        self.policy.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        return save_path

    def remove_model(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        logging.info('remove model checkpoint of %s', path)

    def policy_evaluation(self, seed=0):
        with open('data/self-generation/train.pickle', 'rb') as f:
            self.reader.train = pickle.load(f)

        scores = np.load('data/scores/cum_scores_iter_0.npy', allow_pickle=True).tolist()
        all_batches = self.reader.get_transition_batches('train', scores, state_until_db=True)

        optimizer, scheduler, critic_optimizer = self.get_optimizers()
        self.target_critic.eval()

        logging.info("***** Policy Evaluation *****")
        logging.info("  Num Epochs = %d", cfg.epoch_num)

        if not os.path.exists('policy_evaluation_weight/'):
            os.makedirs('policy_evaluation_weight/')

        for epoch in range(5):
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

                if skip_condition:
                    continue

                self.critic.train()

                inputs = add_torch_transition_input(self, inputs)

                hidden = self.policy(inputs['sa_tensor'], get_hidden=True)
                values = self.critic(hidden)

                next_hidden = self.policy(inputs['next_sa_tensor'], get_hidden=True)
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

            if not os.path.exists('policy_evaluation_weight_wbc'):
                os.mkdir('policy_evaluation_weight_wbc')
            torch.save(self.critic, 'policy_evaluation_weight_wbc/critic_seed_{}'.format(seed))

    def generate_rewards(self):
        with open('data/self-generation/train.pickle', 'rb') as f:
            data = pickle.load(f)

        successes, matches, scores = self.evaluator.training_data_metric(data, self.tokenizer)

        if not os.path.exists('data/scores/'):
            os.makedirs('data/scores/')
        np.save('data/scores/cum_successes_iter_0.npy', np.array(successes))
        np.save('data/scores/cum_matches_iter_0.npy', np.array(matches))
        np.save('data/scores/cum_scores_iter_0.npy', np.array(scores))

        print('Train Dataset')
        print('Success: {:.4f} / Match: {:.4f}'.format(np.mean(successes), np.mean(matches)))

    def validate(self, data='dev', do_test=False):
        # predict one dialog/ one turn at a time
        self.policy.eval()
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
                    inputs = add_torch_input_eval(self, inputs)
                    # fail to generate new tokens, if max_length not set
                    context_length = len(inputs['context'])
                    if cfg.use_true_curr_bspn:  # generate act, response
                        max_len = 60
                        if not cfg.use_true_curr_aspn:
                            max_len = 80

                        outputs = self.policy.generate(input_ids=inputs['context_tensor'],
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
                        outputs = self.policy.generate(input_ids=inputs['context_tensor'],
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
                        outputs_db = self.policy.generate(input_ids=inputs['context_tensor_db'],
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
                    pv_turn['db'] = turn['db'] if cfg.use_true_curr_bspn else db
                    pv_turn['aspn'] = turn['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']

                result_collection.update(
                    self.reader.inverse_transpose_turn(dialog))

        logging.info("inference time: {:.2f} min".format((time.time() - btm) / 60))
        # score
        btm = time.time()
        results, _ = self.reader.wrap_result_lm(result_collection)
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

        if cfg.use_true_curr_aspn:  # only predict resp
            decoded['resp'] = generated[: eos_r_idx + 1]
        else:  # predicted aspn, resp
            eos_a_idx = generated.index(eos_a_id)
            decoded['aspn'] = generated[: eos_a_idx + 1]
            decoded['resp'] = generated[eos_a_idx + 1: eos_r_idx + 1]
        return decoded

    def decode_generated_bspn(self, generated):
        eos_b_id = self.tokenizer.encode(['<eos_b>'])[0]
        if eos_b_id in generated:
            eos_b_idx = generated.index(eos_b_id)
        else:
            eos_b_idx = len(generated) - 1
        return generated[: eos_b_idx + 1]

    def add_torch_input_advantage_eval(self, inputs):
        # inputs: context, state
        if len(inputs['context']) > 1024:
            inputs['context_tensor'] = torch.tensor([inputs['context'][-1024:]]).to(self.device)
        else:
            inputs['context_tensor'] = torch.tensor([inputs['context']]).to(self.device)

        if len(inputs['state']) > 1024:
            inputs['state_tensor'] = torch.tensor([inputs['state'][-1024:]]).to(self.device)
        else:
            inputs['state_tensor'] = torch.tensor([inputs['state']]).to(self.device)

        # inputs['context_tensor'] = torch.tensor([inputs['context']]).to(self.device)
        # inputs['state_tensor'] = torch.tensor([inputs['state']]).to(self.device)
        return inputs