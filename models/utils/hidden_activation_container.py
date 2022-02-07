"""
A container for the activations of the f2 dense layer.

2021 Vadym Gryshchuk (vadym.gryshchuk@protonmail.com)
"""

import torch


class HiddenActivationContainer(object):

    def __init__(self, device, n_neurons, habituation_decay_rate):
        # Parameters:
        self.habituation_counter = torch.ones(n_neurons).to(device)
        self.habituation_decay_rate = habituation_decay_rate

        # Metrics:
        self.hidden_activations = []
        self.norm_term = 0.0

    def add_hidden_activations(self, hidden_activations):
        self.hidden_activations.append(hidden_activations)

    def reset_hidden_activations(self):
        self.hidden_activations = []

    def reset_norm(self):
        self.norm_term = 0.0

    def calculate_l1_slowness_term(self):
        """
        Calculates L1-norm value over the difference between previous activations and current.
        :return: L1-norm term.
        """

        for i in range(0, len(self.hidden_activations)):
            if i == 0:
                self.norm_term += torch.sum(torch.square(self.hidden_activations[i]))
            self.norm_term += torch.sum(torch.square(self.hidden_activations[i - 1] - self.hidden_activations[i]))

    def update_habituation_counter(self, neuron_indices):
        for i in neuron_indices:
            self.habituation_counter[i] += self.habituation_decay_rate * (1 - self.habituation_counter[i]) - \
                                           self.habituation_decay_rate  # 0.001 or 0.1

    def get_habituation_counter(self):
        """
        Returns a habituation counter.
        :return: a habituation counter tensor.
        """
        return self.habituation_counter

    def get_norm_term(self):
        """
        Returns an L-norm loss term.
        :return: an L-norm loss term.
        """
        return self.norm_term
