from Code.dir_definitions import *
from Code.plotter import get_ser_data, plot_ser_by_block_index, plot_ser_by_snr
from Code.trainer import Trainer


def execute_and_plot(model_name, detector_method, self_supervised, all_curves, current_params, run_over):
    method_name = model_name + "_" + detector_method
    trainer = Trainer(
                    model_name=model_name,
                    detector_method=detector_method,
                    self_supervised=self_supervised,
                    weights_dir=os.path.join(WEIGHTS_DIR,
                    f'{method_name}_training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1_{HYPERPARAMS_DICT["channel_coefficients"]}'),
                    **HYPERPARAMS_DICT)

    ser = get_ser_data(trainer, run_over=run_over, method_name=method_name + '_' + current_params)
    all_curves.append((ser, model_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


HYPERPARAMS_DICT = {
                    'noisy_est_var': 0,
                    'fading_taps_type': 1,  # 1 / 2  for time decay only
                    'fading_in_channel': True,
                    'fading_in_decoder': True,
                    'gamma': 0.2,
                    'channel_type': 'ISI_AWGN',
                    'val_frames': 12,  # up to 12 for cost2100
                    'subframes_in_frame': 25,  # up to 25 for cost2100
                    'self_supervised_iterations': 200,
                    'ser_thresh': 0.02,  # ser threshold for online training
                    'train_minibatch_num': 25,  # 25
                    }


if __name__ == '__main__':
    # main flags
    run_over = 0  # 0 - load plots from previous runs if exists / 1 - load trained weights and start online evaluation / 2 - clear all and start training  from scratch
    plot_by_block = False  # False / True either plot by SNR or by block index
    block_length = 120     # determine the transmission length
    channel_coefficients = 'cost2100'  # 'time_decay' / 'cost2100'
    n_symbol = 2
    snr_start, snr_end = 7, 15

    # deep learning models list 'ADNN', 'Sionna', 'SionnaPlus', 'Transformer', 'LSTM', 'ViterbiNet'
    models_list = ['ADNN', 'Sionna', 'SionnaPlus', 'Transformer', 'LSTM', 'ViterbiNet']
    detector_method = 'ModelBased'  # ModelBased / EndToEnd / Statistical
    self_supervised = True  # True / False for online evaluation enablement

    all_curves = []

    for snr in range(snr_start, snr_end+1):
        print(snr, block_length, n_symbol)

        HYPERPARAMS_DICT['n_symbols'] = n_symbol
        HYPERPARAMS_DICT['curr_SNR'] = snr
        HYPERPARAMS_DICT['val_block_length'] = block_length
        HYPERPARAMS_DICT['train_block_length'] = block_length
        HYPERPARAMS_DICT['fading_in_channel'] = True if channel_coefficients == 'time_decay' else False
        HYPERPARAMS_DICT['channel_coefficients'] = channel_coefficients

        current_params = HYPERPARAMS_DICT['channel_coefficients'] + '_' + str(HYPERPARAMS_DICT['curr_SNR']) + '_' + \
                         str(HYPERPARAMS_DICT['val_block_length']) + '_' + str(HYPERPARAMS_DICT['n_symbols'])

        for model in models_list:
            execute_and_plot(model, detector_method, self_supervised, all_curves, current_params, run_over)

        execute_and_plot('ClassicViterbi', 'Statistical', False, all_curves, current_params, run_over)  # Classic Viterbi Alg with Perfect-CSI

        if plot_by_block:
            plot_ser_by_block_index(all_curves, block_length, n_symbol, snr)

    snr_values = [s for s in range(snr_start, snr_end+1)]
    if not plot_by_block:
        plot_ser_by_snr(all_curves, snr_values)




