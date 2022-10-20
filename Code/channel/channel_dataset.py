from Code.channel.channel_estimation import estimate_channel
from Code.channel.modulator import BPSKModulator
from Code.channel.channel import ISIAWGNChannel
from Code.ecc.rs_main import encode
from torch.utils.data import Dataset
from numpy.random import mtrand
from typing import Tuple, List
import numpy as np
import torch

import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation to draw minibatches of channel words and transmitted
    """

    def __init__(self, channel_type: str,
                 block_length: int,
                 transmission_length: int,
                 words: int,
                 memory_length: int,
                 channel_coefficients: str,
                 random: mtrand.RandomState,
                 word_rand_gen: mtrand.RandomState,
                 noisy_est_var: float,
                 fading_taps_type: int,
                 use_ecc: bool,
                 n_symbols: int,
                 fading_in_channel: bool,
                 fading_in_decoder: bool,
                 phase: str):

        self.block_length = block_length
        self.transmission_length = transmission_length
        self.word_rand_gen = word_rand_gen if word_rand_gen else np.random.RandomState()
        self.random = random if random else np.random.RandomState()
        self.channel_type = channel_type
        self.words = words
        self.memory_length = memory_length
        self.channel_coefficients = channel_coefficients
        self.noisy_est_var = noisy_est_var
        self.fading_taps_type = fading_taps_type
        self.fading_in_channel = fading_in_channel
        self.fading_in_decoder = fading_in_decoder
        self.n_symbols = n_symbols
        self.phase = phase
        if use_ecc:
            self.encoding = lambda b: encode(b, self.n_symbols)
        else:
            self.encoding = lambda b: b

    def get_snr_data(self, snr: float, gamma: float, database: list):
        # if database is None:
        #     database = []
        b_full = np.empty((0, self.block_length))
        y_full = np.empty((0, self.transmission_length))
        if self.phase == 'val':
            index = 0
        else:
            index = 0  # random.randint(0, 1e6)
        # accumulate words until reaches desired number
        while y_full.shape[0] < self.words:
            # generate word
            b = self.word_rand_gen.randint(0, 2, size=(1, self.block_length))
            # encoding - errors correction Code
            c = self.encoding(b).reshape(1, -1)
            # add zero bits
            padded_c = np.concatenate([c, np.zeros([c.shape[0], self.memory_length])], axis=1)
            # transmit
            h = estimate_channel(self.memory_length, gamma,
                                 channel_coefficients=self.channel_coefficients,
                                 noisy_est_var=self.noisy_est_var,
                                 fading=self.fading_in_channel if self.phase == 'val' else self.fading_in_decoder,
                                 index=index,
                                 fading_taps_type=self.fading_taps_type)
            y = self.transmit(padded_c, h, snr)
            # accumulate
            b_full = np.concatenate((b_full, b), axis=0)
            y_full = np.concatenate((y_full, y), axis=0)
            index += 1

        database.append((b_full, y_full))

    def transmit(self, c: np.ndarray, h: np.ndarray, snr: float):
        if self.channel_type == 'ISI_AWGN':
            # modulation
            s = BPSKModulator.modulate(c)
            # transmit through noisy channel
            y = ISIAWGNChannel.transmit(s=s, random=self.random, h=h, snr=snr, memory_length=self.memory_length)
        else:
            raise Exception('No such channel defined!!!')
        return y

    def __getitem__(self, snr_list: List[float], gamma: float) -> Tuple[torch.Tensor, torch.Tensor]:
        database = []
        # do not change max_workers
        [self.get_snr_data(snr, gamma, database) for snr in snr_list]
        b, y = (np.concatenate(arrays) for arrays in zip(*database))
        b, y = torch.Tensor(b).to(device=device), torch.Tensor(y).to(device=device)
        return b, y

    def __len__(self):
        return self.transmission_length
