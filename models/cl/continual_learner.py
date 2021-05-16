"""
This file is copied from
https://github.com/GMvandeVen/brain-inspired-replay/tree/master/models/cl

Adaptations of the code are performed by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

"""

import abc

import numpy as np
import torch
from torch import nn


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract  module to add continual learning capabilities to a classifier.

    Adds methods/attributes for "context-dependent gating" (XdG), "elastic weight consolidation" (EWC),
    "synaptic intelligence" (SI) and "generative replay" (GR) to its subclasses.'''

    def __init__(self):
        super().__init__()

        # XdG:
        self.mask_dict = None  # -> <dict> with task-specific masks for each hidden fully-connected layer
        self.excit_buffer_list = []  # -> <list> with excit-buffers for all hidden fully-connected layers

        # SI:
        self.si_c = 0  # -> hyperparam: how strong to weigh SI-loss ("regularisation strength")
        self.epsilon = 0.1  # -> dampening parameter: bounds 'omega' when squared parameter-change goes to 0

        # Replay:
        self.replay_targets = "hard"  # should distillation loss be used? (hard|soft)
        self.KD_temp = 2.  # temperature for distillation loss

        # Habituation
        self.habituation = False
        self.habituation_decay_rate = 0
        self.slowness = False

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x):
        pass

    # ----------------- XdG-specifc functions -----------------#

    def define_XdGmask(self, gating_prop, n_tasks):
        '''Define task-specific masks, by randomly selecting [gating_prop]% of nodes per hidden fc-layer for each task.

        [gating_prop]   <num>, between 0 and 1, proportion of nodes to be gated
        [n_tasks]       <int>, total number of tasks'''

        mask_dict = {}
        excit_buffer_list = []
        for task_id in range(n_tasks):
            mask_dict[task_id + 1] = {}
            for i in range(self.fcE.layers):
                layer = getattr(self.fcE, "fcLayer{}".format(i + 1)).linear
                if task_id == 0:
                    excit_buffer_list.append(layer.excit_buffer)
                n_units = len(layer.excit_buffer)
                gated_units = np.random.choice(n_units, size=int(gating_prop * n_units), replace=False)
                mask_dict[task_id + 1][i] = gated_units
        self.mask_dict = mask_dict
        self.excit_buffer_list = excit_buffer_list

    def apply_XdGmask(self, task):
        '''Apply task-specific mask, by setting activity of pre-selected subset of nodes to zero.

        [task]   <int>, starting from 1'''

        assert self.mask_dict is not None
        torchType = next(self.parameters()).detach()

        # Loop over all buffers for which a task-specific mask has been specified
        for i, excit_buffer in enumerate(self.excit_buffer_list):
            gating_mask = np.repeat(1., len(excit_buffer))
            gating_mask[self.mask_dict[task][i]] = 0.  # -> find task-specifc mask
            excit_buffer.set_(torchType.new(gating_mask))  # -> apply this mask

    def reset_XdGmask(self):
        '''Remove task-specific mask, by setting all "excit-buffers" to 1.'''
        torchType = next(self.parameters()).detach()
        for excit_buffer in self.excit_buffer_list:
            gating_mask = np.repeat(1., len(excit_buffer))  # -> define "unit mask" (i.e., no masking at all)
            excit_buffer.set_(torchType.new(gating_mask))  # -> apply this unit mask

    # ------------- "Intelligent Synapses"-specifc functions -------------#

    def update_omega(self, W, epsilon):
        '''After completing training on a task, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n] / (p_change ** 2 + epsilon)
                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store these new values in the model
                self.register_buffer('{}_SI_prev_task'.format(n), p_current)
                self.register_buffer('{}_SI_omega'.format(n), omega_new)

    def surrogate_loss(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self, '{}_SI_prev_task'.format(n))
                    omega = getattr(self, '{}_SI_omega'.format(n))
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p - prev_values) ** 2).sum())
            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())
