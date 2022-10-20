import torch.nn as nn
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

START_PADDING_VALUE = -100


class Detector(nn.Module):
    def __init__(self, model, detector_method):
        super(Detector, self).__init__()
        self.model = model
        self.detector_method = detector_method
        print(f'{self.model.__class__.__name__}_{self.detector_method}')

        self.transition_table = None

        self.initialize_transition_table()

    def initialize_transition_table(self):
        if self.detector_method != 'EndToEnd':
            transition_table_array = np.concatenate([np.arange(self.model.n_classes), np.arange(self.model.n_classes)]).reshape(self.model.n_classes, 2)
            self.transition_table = torch.Tensor(transition_table_array).to(device)

    def acs_block(self, in_prob: torch.Tensor, llrs: torch.Tensor) -> [
        torch.Tensor, torch.LongTensor]:
        """
        Viterbi ACS (Add-Compare-Select) block
        :param in_prob: last stage probabilities, [batch_size,n_states]
        :param llrs: edge probabilities, [batch_size,1]
        :return: current stage probabilities, [batch_size,n_states]
        """
        transition_ind = self.transition_table.reshape(-1).repeat(in_prob.size(0)).long()
        batches_ind = torch.arange(in_prob.size(0)).repeat_interleave(2 * self.model.n_classes)
        trellis = (in_prob + llrs)[batches_ind, transition_ind]
        reshaped_trellis = trellis.reshape(-1,  self.model.n_classes, 2)
        return torch.min(reshaped_trellis, dim=2)

    def forward(self, input_: torch.Tensor, phase: str):
        batch_size, transmission_length = input_.size(0), input_.size(1)
        padded_input = torch.nn.functional.pad(input_, [0, self.model.input_size - 1, 0, 0], value=START_PADDING_VALUE)
        sequence_input = torch.cat([torch.roll(padded_input.unsqueeze(1), i, 2) for i in range(self.model.input_size - 1, -1, -1)], dim=1)
        sequence_input = sequence_input.transpose(1, 2)[:, :transmission_length].to(device)

        estimation = self.model(sequence_input)

        if self.detector_method == 'ModelBased' or self.detector_method == 'Statistical':  # Viterbi Algorithm
            priors = estimation
            if phase == 'val':
                # initialize input probabilities
                in_prob = torch.zeros([input_.shape[0], self.model.n_classes]).to(device)
                reconstructed_word = torch.zeros(input_.shape).to(device)
                for i in range(transmission_length):
                    # get the lsb of the state
                    reconstructed_word[:, i] = torch.argmin(in_prob, dim=1) % 2
                    # run one Viterbi stage
                    out_prob, _ = self.acs_block(in_prob, -priors[:, i])
                    # update in-probabilities for next layer
                    in_prob = out_prob
                return reconstructed_word
            else:
                return priors
        elif self.detector_method == 'EndToEnd':
            if phase == 'val':
                return torch.argmax(estimation, dim=2).float()
            else:
                return estimation


