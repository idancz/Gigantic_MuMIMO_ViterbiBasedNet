from Code.trainer import Trainer
from Code.dir_definitions import FIGURES_DIR, PLOTS_DIR
import datetime
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import math
import matplotlib as mpl

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

MIN_BER_COEF = 0.2
MARKER_EVERY = 20

COLORS_DICT = {'ViterbiNet': 'green',
               'LSTM': 'green',
               'ClassicViterbi': 'black',
               'Sionna': 'blue',
               'SionnaPlus': 'purple',
               'SionnaAdd': 'red',
               'SionnaSkip': 'gray',
               'ADNN': 'red',
               'Transformer': 'orange',
               }

MARKERS_DICT = {'ViterbiNet': 'd',
                'LSTM': 'd',
                'ClassicViterbi': 'o',
                'Sionna': 'x',
                'SionnaPlus': 'x',
                'SionnaAdd': 'x',
                'SionnaSkip': 'x',
                'ADNN': 'd',
                'Transformer': 'x',
                }

LINESTYLES_DICT = {'ViterbiNet': 'solid',
                   'LSTM': 'dotted',
                   'ClassicViterbi': 'solid',
                   'Sionna': 'solid',
                   'SionnaPlus': 'solid',
                   'SionnaAdd': 'solid',
                   'SionnaSkip': 'solid',
                   'ADNN': 'dotted',
                   'Transformer': 'solid',
                   }

METHOD_NAMES = {'ViterbiNet': 'ViterbiNet',
                'LSTM': 'LSTM',
                'ClassicViterbi': 'Viterbi, perfect CSI',
                'Sionna': 'Sionna',
                'SionnaPlus': 'SionnaPlus',
                'SionnaAdd': 'SionnaAdd',
                'SionnaSkip': 'SionnaSkip',
                'ADNN': 'ADNN',
                'Transformer': 'Transformer',
                }


def get_ser_data(trainer: Trainer, run_over: bool, method_name: str):
    print(method_name)
    # set the path to saved plot results for a single method (so we do not need to run anew each time)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    file_name = '_'.join([method_name, str(trainer.channel_type)])
    plots_path = os.path.join(PLOTS_DIR, file_name + '.pkl')
    print(plots_path)
    # if plot already exists, and the run_over flag is false - load the saved plot
    if os.path.isfile(plots_path) and not run_over:
        print("Loading plots")
        ser_total = load_pkl(plots_path)
    else:
        # otherwise - run again
        print("calculating fresh")
        ser_total = trainer.run(run_over)
        save_pkl(plots_path, ser_total)
    print(np.mean(ser_total))
    return ser_total


def plot_summary_table(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], models_list: list, snr_values: list):
    df = pd.DataFrame(all_curves, columns=['BER', 'NET', 'InfoBits', 'Modulation'])
    results = np.zeros((len(snr_values), len(models_list)))
    i, j = 0, 0
    for snr in snr_values:
        j = 0
        for model in models_list:
            model_ber = df[df.NET == model]
            model_ber.index = snr_values
            results[i, j] = model_ber[['BER']].loc[snr].mean(axis=0, skipna=True).mean()
            j += 1
        i += 1
    res = pd.DataFrame(results, columns=models_list)
    res.index = snr_values
    res.index.name = 'SNR'
    title = '##############################> - SNR  Summary Table - <##############################'
    print("______________________________________________________________________________________")
    print(title)
    print("______________________________________________________________________________________")
    print(res)


def plot_ser_by_block_index(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], val_block_length: int,
                               n_symbol: int, snr: float):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    plt.figure()
    min_block_ind = math.inf
    min_ber = math.inf
    max_block_ind = -math.inf
    # iterate all curves, plot each one
    for ser, method_name, _, _ in all_curves:
        print(method_name)
        print(len(ser))
        block_range = np.arange(1, len(ser) + 1)
        key = method_name.split(' ')[0]
        agg_ser = (np.cumsum(ser) / np.arange(1, len(ser) + 1))
        plt.plot(block_range, agg_ser,
                 label=METHOD_NAMES[key],
                 color=COLORS_DICT[key], marker=MARKERS_DICT[key],
                 linestyle=LINESTYLES_DICT[key], linewidth=2.2, markevery=MARKER_EVERY)
        min_block_ind = block_range[0] if block_range[0] < min_block_ind else min_block_ind
        max_block_ind = block_range[-1] if block_range[-1] > max_block_ind else max_block_ind
        min_ber = agg_ser[-1] if agg_ser[-1] < min_ber else min_ber
    plt.ylabel('Coded BER')
    plt.xlabel('Block Index')
    plt.xlim([min_block_ind - 0.1, max_block_ind + 0.1])
    plt.ylim(bottom=MIN_BER_COEF * min_ber)
    plt.yscale('log')
    plt.legend(loc='upper left', prop={'size': 15})
    plt.savefig(
        os.path.join(FIGURES_DIR, folder_name,
                     f'SNR {snr}, Block Length {val_block_length}, Error symbols {n_symbol}.png'),
        bbox_inches='tight')
    plt.show()


def plot_ser_by_snr(all_curves: List[Tuple[np.ndarray, np.ndarray, str]], snr_values: List[float]):
    # path for the saved figure
    current_day_time = datetime.datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    plt.figure()
    names = []
    for i in range(len(all_curves)):
        if all_curves[i][1] not in names:
            names.append(all_curves[i][1])

    for method_name in names:
        mean_sers = []
        key = method_name.split(' ')[0]
        for ser, cur_name, val_block_length, n_symbol in all_curves:
            mean_ser = np.mean(ser)
            if cur_name != method_name:
                continue
            mean_sers.append(mean_ser)
        plt.plot(snr_values, mean_sers, label=METHOD_NAMES[key],
                 color=COLORS_DICT[key], marker=MARKERS_DICT[key],
                 linestyle=LINESTYLES_DICT[key], linewidth=2.2)

    plt.xticks(snr_values, snr_values)
    plt.xlabel('SNR [dB]')
    plt.ylabel('Coded BER')
    plt.grid(which='both', ls='--')
    plt.legend(loc='lower left', prop={'size': 15})
    plt.yscale('log')
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'coded_ber_versus_block_length.png'),
                bbox_inches='tight')
    plt.show()


def save_pkl(pkls_path: str, array: np.ndarray):
    output = open(pkls_path, 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str):
    output = open(pkls_path, 'rb')
    return pkl.load(output)
