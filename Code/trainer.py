from Code.models import ClassicViterbi, ViterbiNet, LSTM, SionnaNeuralReceiver, SionnaSkip, SionnaViterbiPlus, SionnaViterbiAdd, ECC_Transformer, ADNN
from Code.detector import Detector
from Code.channel.channel_dataset import ChannelModelDataset
from Code.ecc.rs_main import decode, encode
from Code.dir_definitions import CONFIG_PATH, WEIGHTS_DIR
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from torch.optim import RMSprop, Adam, SGD
from typing import Tuple, Union
from shutil import copyfile
from time import time
import numpy as np
import yaml
import torch
import os
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


INPUT_SIZE = 4    # input rolling number
N_DIM = 32

N_HEADS = 8       # for Transformers model

HIDDEN_SIZE = 256  # for LSTM model
NUM_LAYERS = 2

N_CLASSES = 2      # for EndToEnd method


class Trainer(object):
    def __init__(self, config_path=None, **kwargs):

        # general
        self.run_name = None

        # Code parameters
        self.n_symbols = None

        # channel
        self.memory_length = None
        self.channel_type = None
        self.channel_coefficients = None
        self.noisy_est_var = None
        self.fading_in_channel = None
        self.fading_in_decoder = None
        self.fading_taps_type = None
        self.subframes_in_frame = None
        self.gamma = None
        self.curr_SNR = None

        # validation hyperparameters
        self.val_block_length = None
        self.val_frames = None

        # training hyperparameters
        self.train_block_length = None
        self.train_frames = None
        self.train_minibatch_num = None
        self.train_minibatch_size = None
        self.lr = None  # learning rate
        self.loss_type = None
        self.optimizer_type = None

        # self-supervised online training
        self.self_supervised = None
        self.self_supervised_iterations = None
        self.ser_thresh = None

        # seed
        self.noise_seed = None
        self.word_seed = None

        # weights dir
        self.weights_dir = None

        # detector model
        self.model_name = None
        self.detector_method = None
        self.detector = None

        # if any kwargs are passed, initialize the dict with them
        self.initialize_by_kwargs(**kwargs)

        # initializes all none parameters above from config
        self.param_parser(config_path)

        # initializes word and noise generator from seed
        self.rand_gen = np.random.RandomState(self.noise_seed)
        self.word_rand_gen = np.random.RandomState(self.word_seed)
        self.n_states = 2 ** self.memory_length

        # initialize matrices, datasets and detector
        self.initialize_channel_data()
        self.initialize_detector()

        # calculate data subframes indices. We will calculate ser only over these values.
        self.data_indices = torch.Tensor(list(filter(lambda x: x % self.subframes_in_frame != 0,
                                                     [i for i in
                                                      range(self.val_frames * self.subframes_in_frame)]))).long()

    def initialize_by_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def param_parser(self, config_path: str):
        """
        Parse the config, load all attributes into the trainer
        :param config_path: path to config
        """
        if config_path is None:
            config_path = CONFIG_PATH

        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # set attribute of Trainer with every config item
        for k, v in self.config.items():
            try:
                if getattr(self, k) is None:
                    setattr(self, k, v)
            except AttributeError:
                pass

        if self.weights_dir is None:
            self.weights_dir = os.path.join(WEIGHTS_DIR, self.run_name)
            if not os.path.exists(self.weights_dir) and len(self.weights_dir):
                os.makedirs(self.weights_dir)
                # save config in output dir
                copyfile(config_path, os.path.join(self.weights_dir, "configuration.yaml"))

    def get_name(self):
        return self.__name__()

    def initialize_detector(self):
        if self.detector_method != 'EndToEnd':
            n_classes = self.n_states
        else:
            n_classes = N_CLASSES
        if self.detector_method == 'Statistical':
            self.self_supervised = False
        models = {
            'ClassicViterbi': lambda: ClassicViterbi(n_classes=n_classes,
                                   memory_length=self.memory_length,
                                   gamma=self.gamma,
                                   val_words=self.val_frames * self.subframes_in_frame,
                                   channel_type=self.channel_type,
                                   noisy_est_var=self.noisy_est_var,
                                   fading=self.fading_in_decoder,
                                   fading_taps_type=self.fading_taps_type,
                                   channel_coefficients=self.channel_coefficients),
            'ViterbiNet': lambda: ViterbiNet(input_size=1, n_classes=self.n_states),
            'LSTM': lambda: LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, n_classes),
            'ADNN': lambda: ADNN(input_size=INPUT_SIZE, dim=N_DIM, n_classes=n_classes),
            'Sionna': lambda: SionnaNeuralReceiver(input_size=1, n_input_channels=1, n_output_channels=N_DIM, n_classes=n_classes),
            'SionnaPlus': lambda: SionnaViterbiPlus(input_size=1, n_input_channels=1, n_output_channels=N_DIM, n_classes=n_classes),
            'SionnaAdd': lambda: SionnaViterbiAdd(input_size=1, n_input_channels=1, n_output_channels=N_DIM, n_classes=n_classes),
            'SionnaSkip': lambda: SionnaSkip(input_size=1, n_input_channels=1, n_output_channels=N_DIM, n_classes=n_classes),
            'Transformer': lambda: ECC_Transformer(INPUT_SIZE, N_DIM, N_HEADS, NUM_LAYERS, n_classes),
        }
        selected_model = models[self.model_name]().to(device)
        self.detector = Detector(selected_model, self.detector_method)

    # configurate the optimization algorithms
    def config_optimizer(self):
        """
        Sets up the optimizer and loss criterion
        """
        if self.optimizer_type == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.detector.model.parameters()),
                                  lr=self.lr)
        elif self.optimizer_type == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.detector.model.parameters()),
                                 lr=self.lr)
        elif self.optimizer_type == 'RMSprop':
            self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.detector.model.parameters()),
                                     lr=self.lr)
        else:
            raise NotImplementedError("No such optimizer implemented!!!")

    # configurate the loss function
    def config_criterion(self):
        if self.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(device)
        else:
            raise NotImplementedError("Not supported such loss function!!!")

    def initialize_channel_data(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.block_lengths = {'train': self.train_block_length, 'val': self.val_block_length}
        self.frames_per_phase = {'train': self.train_frames, 'val': self.val_frames}
        self.transmission_lengths = {'train': self.train_block_length + 8 * self.n_symbols,
                                     'val': self.val_block_length + 8 * self.n_symbols}
        self.channel_dataset = {
            phase: ChannelModelDataset(channel_type=self.channel_type,
                                       block_length=self.block_lengths[phase],
                                       transmission_length=self.transmission_lengths[phase],
                                       words=self.frames_per_phase[phase] * self.subframes_in_frame,
                                       memory_length=self.memory_length,
                                       channel_coefficients=self.channel_coefficients,  # time_decay / cost2100
                                       random=self.rand_gen,
                                       word_rand_gen=self.word_rand_gen,
                                       noisy_est_var=self.noisy_est_var,
                                       use_ecc=True,
                                       n_symbols=self.n_symbols,
                                       fading_taps_type=self.fading_taps_type,
                                       fading_in_channel=self.fading_in_channel,
                                       fading_in_decoder=self.fading_in_decoder,
                                       phase=phase) for phase in ['train', 'val']}

    def run(self, run_over, num_of_rep=1) -> np.ndarray:
        """
        Train and evaluation in a word-by-word way
        """
        self.load_train_weights(run_over)
        return self.online_evaluation(num_of_rep)

    def train(self):
        """
        Main training loop. Runs in minibatches.
        Evaluates performance over validation SNR.
        Saves weights given the best validation SER result.
        """
        if self.detector_method == 'Statistical':
            raise NotImplementedError("No training implemented for Statistical decoder!!!")
        self.config_optimizer()
        self.config_criterion()

        print(f'Start training SNR - {self.curr_SNR}, Gamma - {self.gamma}, Channel Cost - {self.channel_coefficients}')
        best_ser = math.inf
        for minibatch in range(1, self.train_minibatch_num + 1): # batches loop
            # draw words
            transmitted_words, received_words = self.channel_dataset['train'].__getitem__(snr_list=[self.curr_SNR], gamma=self.gamma)

            # transmitted_words = torch.cat([torch.Tensor(
            #     encode(transmitted_word.int().cpu().numpy(), self.n_symbols).reshape(1, -1)).to(device)
            #                                for transmitted_word in transmitted_words], dim=0)

            # run training loops
            current_loss = 0
            for i in range(self.train_frames * self.subframes_in_frame):  # train over one minibatch
                predictions = self.detector(received_words[i].reshape(1, -1), 'train')  # pass through detector
                current_loss += self.backpropagation(predictions, transmitted_words[i].reshape(1, -1))   # calculate loss and update weights

            # evaluate performance - Symbol Error Rate
            ser = self.evaluate()
            print(f'Minibatch {minibatch} | Train Loss {current_loss} | Validation SER - {ser}')
            if ser < best_ser:
                self.save_weights(current_loss)  # save best weights
                best_ser = ser

        print(f'Best Validation SER - {best_ser} (saved)')
        print('*' * 50)

    def backpropagation(self, predictions: torch.Tensor, transmitted_words: torch.Tensor):
        # calculate loss
        loss = self.calculate_loss(predictions=predictions, transmitted_words=transmitted_words)
        # if loss is Nan inform the user
        if torch.sum(torch.isnan(loss)):
            print('loss value is Nan')
            return np.nan
        current_loss = loss.item()
        # back propagation
        for param in self.detector.model.parameters():
            param.grad = None
        loss.backward()
        self.optimizer.step()
        return current_loss

    # calculate train loss
    def calculate_loss(self, predictions: torch.Tensor, transmitted_words: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        Works for EndToEnd / ModelBased / Statistical methodologies
        :param predictions: [1,transmission_length,n_states], each element is a probability
        :param transmitted_words: [1, transmission_length]
        :return: loss value
        """
        if self.detector_method != "EndToEnd":
            gt_labels = self.calculate_states(transmitted_words)
            predictions = predictions.reshape(-1, self.n_states)
        else:
            gt_labels = transmitted_words.long().reshape(-1)
            predictions = predictions.reshape(-1, 2)
        gt_labels_batch, input_batch = self.select_batch(gt_labels, predictions)
        loss = self.criterion(input=input_batch, target=gt_labels_batch)
        return loss

    def calculate_states(self, transmitted_words: torch.Tensor) -> torch.Tensor:
        """
        calculates the state for the transmitted words
        :param transmitted_words: channel transmitted words
        :return: vector of length of transmitted_words with values in the range of 0,1,...,n_states-1
        """
        padded = torch.cat([transmitted_words, torch.zeros([transmitted_words.shape[0], self.memory_length]).to(device)],dim=1)
        unsqueezed_padded = padded.unsqueeze(dim=1)
        blockwise_words = torch.cat([unsqueezed_padded[:, :, i:-self.memory_length + i] for i in range(self.memory_length)], dim=1)
        states_enumerator = (2 ** torch.arange(self.memory_length)).reshape(1, -1).float().to(device)
        gt_states = torch.sum(blockwise_words.transpose(1, 2).reshape(-1, self.memory_length) * states_enumerator, dim=1).long()
        return gt_states

    def evaluate(self) -> float:
        """
        Evaluation at a single snr.
        :return: ser for mini-batch
        """
        # draw words of given gamma for all SNRs
        transmitted_words, received_words = self.channel_dataset['val'].__getitem__(snr_list=[self.curr_SNR], gamma=self.gamma)

        # decode and calculate accuracy
        detected_words = self.detector(received_words, 'val')

        # decode the detected words
        decoded_words = [decode(detected_word, self.n_symbols) for detected_word in detected_words.cpu().numpy()]
        detected_words = torch.Tensor(np.array(decoded_words)).to(device)

        ser, fer, err_indices = self.calculate_error_rates(detected_words[self.data_indices], transmitted_words[self.data_indices])
        return ser

    def calculate_error_rates(self, prediction: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
        """
        Returns the ber,fer and error indices
        """
        prediction = prediction.long()
        target = target.long()
        bits_acc = torch.mean(torch.eq(prediction, target).float()).item()
        all_bits_sum_vector = torch.sum(torch.abs(prediction - target), 1).long()
        frames_acc = torch.eq(all_bits_sum_vector, torch.LongTensor(1).fill_(0).to(device=device)).float().mean().item()
        return max([1 - bits_acc, 0.0]), max([1 - frames_acc, 0.0]), torch.nonzero(all_bits_sum_vector, as_tuple=False).reshape(-1)

    def online_evaluation(self, num_of_rep=10) -> Union[float, np.ndarray]:
        print(f'Start online evaluation')
        if self.self_supervised:
            self.config_optimizer()
            self.config_criterion()
        total_ser = 0
        first_run = True
        for rep in range(0, num_of_rep):
            # draw words of given gamma for all SNRs
            transmitted_words, received_words = self.channel_dataset['val'].__getitem__(snr_list=[self.curr_SNR], gamma=self.gamma)

            # received_words = self.get_overlapping_rx(received_words)
            if first_run:
                ser_by_word = np.zeros(num_of_rep*transmitted_words.shape[0])
                # query for all detected words
                buffer_rx = torch.empty([0, received_words.shape[1]]).to(device)
                buffer_tx = torch.empty([0, received_words.shape[1]]).to(device)
                buffer_ser = torch.empty([0]).to(device)
                first_run = False

            for count, (transmitted_word, received_word) in enumerate(zip(transmitted_words, received_words)):
                transmitted_word, received_word = transmitted_word.reshape(1, -1), received_word.reshape(1, -1)
                # detect
                # self.detector.model.eval()
                detected_word = self.detector(received_word, 'val')
                if count in self.data_indices:
                    # decode
                    decoded_word = [decode(detected_word, self.n_symbols) for detected_word in detected_word.cpu().numpy()]
                    decoded_word = torch.Tensor(np.array(decoded_word)).to(device)
                    # calculate accuracy
                    ser, fer, err_indices = self.calculate_error_rates(decoded_word, transmitted_word)
                    # encode word again
                    decoded_word_array = decoded_word.int().cpu().numpy()
                    encoded_word = torch.Tensor(encode(decoded_word_array, self.n_symbols).reshape(1, -1)).to(device)
                    errors_num = torch.sum(torch.abs(encoded_word - detected_word)).item()
                    print(f'{"*" * 35}\nCurrent word: {rep*transmitted_words.shape[0] + count, ser, errors_num}')
                    total_ser += ser
                    ser_by_word[rep*transmitted_words.shape[0] + count] = ser
                else:
                    print(f'{"*" * 35}\nCurrent word: {rep*transmitted_words.shape[0] + count}, Pilot')
                    # encode word again
                    decoded_word_array = transmitted_word.int().cpu().numpy()
                    encoded_word = torch.Tensor(encode(decoded_word_array, self.n_symbols).reshape(1, -1)).to(device)
                    ser = 0
                    errors_num = 0
                # save the encoded word in the buffer
                if ser <= self.ser_thresh:
                    buffer_rx = torch.cat([buffer_rx, received_word])
                    buffer_tx = torch.cat([buffer_tx,
                                           detected_word.reshape(1, -1) if ser > 0 else
                                           encoded_word.reshape(1, -1)],dim=0)
                    buffer_ser = torch.cat([buffer_ser, torch.FloatTensor([ser]).to(device)])

                if self.self_supervised and ser <= self.ser_thresh:
                    # use last word inserted in the buffer for training
                    self.online_training(buffer_tx[-1].reshape(1, -1), buffer_rx[-1].reshape(1, -1))

                if (count + 1) % 10 == 0:
                    print(f'Self-supervised: {rep*transmitted_words.shape[0] + count + 1}/{transmitted_words.shape[0] * num_of_rep}, Average SER {total_ser / (rep*transmitted_words.shape[0] + count + 1)}')

        total_ser /= (transmitted_words.shape[0] * num_of_rep)
        print(f'Final SER: {total_ser}')
        return ser_by_word

    def online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - train on the detected/re-encoded word only if the ser is below some threshold.
        Start from the saved meta-trained weights.
        :param tx: transmitted word
        :param rx: received word
        """
        # run training loops
        for i in range(self.self_supervised_iterations):
            # calculate soft values
            predictions = self.detector(rx, 'train')
            self.backpropagation(predictions=predictions, transmitted_words=tx)

    def select_batch(self, gt_examples: torch.LongTensor, predictions: torch.Tensor) -> Tuple[
        torch.LongTensor, torch.Tensor]:
        """
        Select a batch from the input and gt labels
        :param gt_examples: training labels
        :param predictions: the soft approximation, distribution over states (per word)
        :return: selected batch from the entire "epoch", contains both labels and the NN soft approximation
        """
        rand_ind = torch.multinomial(torch.arange(gt_examples.shape[0]).float(),
                                     self.train_minibatch_size).long().to(device)
        return gt_examples[rand_ind], predictions[rand_ind]

    def save_weights(self, current_loss: float):
        torch.save({'model_state_dict': self.detector.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': current_loss},
                   os.path.join(self.weights_dir, f'snr_{self.curr_SNR}_gamma_{self.gamma}.pt'))

    def load_train_weights(self, run_over):
        """
        Loads detector's weights defined by the [snr,gamma] from checkpoint, if exists else start training
        """
        if self.detector_method == 'Statistical':
            print('Statistical model without weights!')
        else:
            if run_over > 2 or run_over < 0:
                raise ValueError("run_over value out of range 0 - 2 !!!")
            if os.path.join(self.weights_dir, f'snr_{self.curr_SNR}_gamma_{self.gamma}.pt'):
                print(f'loading model from SNR {self.curr_SNR} and gamma {self.gamma}')
                weights_path = os.path.join(self.weights_dir, f'snr_{self.curr_SNR}_gamma_{self.gamma}.pt')
                if not os.path.isfile(weights_path) or run_over == 2:
                    # if weights do not exist, train on the synthetic channel. Then validate on the test channel.
                    self.fading_taps_type = 1
                    os.makedirs(self.weights_dir, exist_ok=True)
                    self.train()
                    self.fading_taps_type = 2
                checkpoint = torch.load(weights_path)
                try:
                    self.detector.model.load_state_dict(checkpoint['model_state_dict'])
                except Exception:
                    raise ValueError("Wrong run directory!!!")
            else:
                print(f'No checkpoint for SNR {self.curr_SNR} and gamma {self.gamma} in run "{self.run_name}", starting from scratch')



