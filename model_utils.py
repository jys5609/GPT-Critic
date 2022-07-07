import numpy as np
import torch


def add_torch_transition_input(model, inputs):
    sa_tensor = torch.from_numpy(inputs['sa_np']).long()
    sa_tensor = sa_tensor.to(model.device)
    inputs['sa_tensor'] = sa_tensor

    next_sa_tensor = torch.from_numpy(inputs['next_sa_np']).long()
    next_sa_tensor = next_sa_tensor.to(model.device)
    inputs['next_sa_tensor'] = next_sa_tensor

    next_s_tensor = torch.from_numpy(inputs['next_s_np']).long()
    next_s_tensor = next_s_tensor.to(model.device)
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
    mask_sa_tensor = mask_sa_tensor.to(model.device)
    inputs['mask_sa_tensor'] = mask_sa_tensor.unsqueeze(-1)

    mask_next_sa_tensor = torch.from_numpy(mask_next_sa).long()
    mask_next_sa_tensor = mask_next_sa_tensor.to(model.device)
    inputs['mask_next_sa_tensor'] = mask_next_sa_tensor.unsqueeze(-1)

    mask_next_s_tensor = torch.from_numpy(mask_next_s).long()
    mask_next_s_tensor = mask_next_s_tensor.to(model.device)
    inputs['mask_next_s_tensor'] = mask_next_s_tensor.unsqueeze(-1)

    terminal_mask = np.zeros(inputs['next_sa_np'].shape)
    for idx, terminal in enumerate(inputs['terminal']):
        terminal_mask[idx][:] = 1 - terminal
    terminal_mask_tensor = torch.from_numpy(terminal_mask).long()
    terminal_mask_tensor = terminal_mask_tensor.to(model.device)
    inputs['terminal_mask_tensor'] = terminal_mask_tensor.unsqueeze(-1)

    reward = np.zeros(inputs['next_sa_np'].shape)
    for idx, r in enumerate(inputs['r']):
        reward[idx][0] = r
    reward_tensor = torch.from_numpy(reward).long()
    reward_tensor = reward_tensor.to(model.device)
    inputs['reward_tensor'] = reward_tensor.unsqueeze(-1)
    return inputs


def add_torch_input(model, inputs):
    contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
    contexts_tensor = contexts_tensor.to(model.device)
    inputs['contexts_tensor'] = contexts_tensor

    maskings_tensor = torch.from_numpy(inputs['maskings_np']).long()
    maskings_tensor = maskings_tensor.to(model.device)
    inputs['maskings_tensor'] = maskings_tensor
    return inputs


def add_torch_input_eval(model, inputs):
    inputs['context_tensor'] = torch.tensor(
        [inputs['context']]).to(model.device)
    return inputs


def add_torch_input_with_advantage(model, inputs):
    contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
    contexts_tensor = contexts_tensor.to(model.device)
    inputs['contexts_tensor'] = contexts_tensor

    advantage_weights_tensor = torch.from_numpy(inputs['advantage_weights_np'])
    advantage_weights_tensor = advantage_weights_tensor.to(model.device)
    inputs['advantage_weights_tensor'] = advantage_weights_tensor
    return inputs


def add_torch_input_advantage_eval(model, inputs):
    # inputs: context, state
    if len(inputs['context']) > 1024:
        inputs['context_tensor'] = torch.tensor([inputs['context'][-1024:]]).to(model.device)
    else:
        inputs['context_tensor'] = torch.tensor([inputs['context']]).to(model.device)

    if len(inputs['state']) > 1024:
        inputs['state_tensor'] = torch.tensor([inputs['state'][-1024:]]).to(model.device)
    else:
        inputs['state_tensor'] = torch.tensor([inputs['state']]).to(model.device)

    # inputs['context_tensor'] = torch.tensor([inputs['context']]).to(self.device)
    # inputs['state_tensor'] = torch.tensor([inputs['state']]).to(self.device)
    return inputs