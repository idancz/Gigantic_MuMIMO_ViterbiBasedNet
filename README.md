# Gigantic MU-MIMO: Toward Channel Statistics Independence in ML Receivers
Deep Neural Networks receivers based Viterbi algorithm for Maximum Likelihood priors-learning in digital communication 

## 1. Introduction   
1.1.  In this project, we explored, extended, and implemented a Viterbi-model-based approaches and DNN architectures for learning the underlying statistics of wireless fading channel communication which obeys a Markovian stochastic input-output relationship ,based on the paper: [“ViterbiNet: A Deep Learning Based Viterbi Algorithm for Symbol Detection” by Nir Shlezinger, Nariman Farsad, Yonina C. Eldar, and Andrea J. Goldsmith](https://arxiv.org/abs/2203.14359)  
## 2. Folder Structure and files Usage
##### note: Files names are marked in ***`italic bold`*** font , while directories are marked in **`bold`** font with '/' suffix.

2.1 **`Code/`** folder - contains all code subdirectories and files implemented in python using `pytorch library`.   
2.1.1. **`channel/`** folder - contains data generation code  
2.1.1.1.	***`channel.py`*** -  contains ISI AWG transmit function  
2.1.1.2.	***`channel_dataset.py`*** – contains the channel data generation class  
2.1.1.3.	***`channel_estimation.py`*** – contains the channel method and costs  
2.1.1.4.	***`modulator.py`*** – contains the BPSK modulation function  
2.1.2.	**`ecc/`** folder - contains the error correction, encoding, decoding files which based on [Reed-Solomon algorithm](https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders)  
2.1.3.	***`dir_definitions.py`*** – includes all project directories and sub-directories  
2.1.4.	***`detector.py`*** – contains the Detector class which responsible for the DNN/Statistical models and methods (ModelBased / EndToEnd / Statistical) including the Viterbi algorithm  
2.1.5.	***`models.py`*** – contains all the project models ADNN / Sionna / SionnaPlus / Transformer / LSTM' / ViterbiNet / ClassicViterbi additional to experimental models, implemented using PyTorch.  
2.1.6.	***`trainer.py`*** – contains the Trainer class which responsible for all training and evaluation flow for a given model including save/load model, configurate loss and optimizer, initialize channel parameters and data, run train loop and backprop, training evaluation, and online evaluation.  
2.1.7.	***`plotter.py`*** – contains the plot functions which creates graphs based on the models results such as “ser by block index” or “ser by snr’ and summary table by SNR function  
2.1.8.	***`configuration.yaml`*** – includes configurable project parameters such as channel parameters /  train and validation hyper-parameters / loss type and available optimizers.  
2.1.9.	***`main.py`*** – responsible for running the project, by looping over SNR list from 7 to 15 dB cross all models and controls all phases of training, evaluation, and graphs using the ```execute_and_plot``` function.
In addition, it contains `HYPERPARAMS_DICT` with configurable parameters and main flags.  
2.1.10.	Important `HYPERPARAMS_DICT` configurable keys:	
>2.1.10.1.	`HYPERPARAMS_DICT [‘val_frames’]` , `HYPERPARAMS_DICT [‘subframes_in_frame’]` - 
 multiplication result of the two determines the Minibatch size during training.  
2.1.10.2.	`HYPERPARAMS_DICT [‘self_supervised_iterations]` - determines the number of self (online) training iterations on the correctly detected block during online evaluation  
2.1.10.3.	`HYPERPARAMS_DICT [‘train_minibatch_num’]` - determines the number of Minibatches during training   

2.1.11.	Important flags in ***`main.py`***
>2.1.11.1.	`run_over` '**0**': loads plots from previous runs, '**1**': loads trained weights and start online evaluation, '**2**': clears all and start training  from scratch.  
2.1.11.2.	`plot_by_block` – '**True**': project will generate `ser by bock index` plot , `**false**`  project will generate `ser by snr` plot.  
2.1.11.3.	`block_length` – determines the transmission length of each block i.e. the number of bits.  
2.1.11.4.	`channel_coefficients` - **time_decay** / **cost2100** determines the channel model type.  
2.1.11.5.	`snr_start`, `snr_end` – determine the range of SNRs  
2.1.11.6.	`models_list` – contains list of all DNN models which implement  detector  
2.1.11.7.	`detector_method` – determine the detector methodology. **ModelBased** for Viterbi based and llr learning / **EndToEnd** for E2E (bit to bit) learning without Viterbi   / **Statistical** used only for the `ClassicViterbi` model which is the statistical Viterbi algorithm with perfect CSI.  
2.1.11.8.	`self_supervised`  - **True/False** for online evaluation enable  

##### Note: every parameter configured in the main.py is the last that counts and overwrites configuration.yaml file values 

2.2.	**`Resources/`** folder - contains the channel coefficients vectors for cost2100 (4 taps, each with 300 blocks).  
2.3.	**`Results/`** folder   
2.3.1.	**`figures/`** folder – contains all the saved graphs images  
2.3.2.	**`plots/`** folder – contains all the saved plots data  
2.3.3.	**`weights/`** folder – contains all the trained model's weights per channel model , modulation order  ,  SNR and gamma.  
2.4.	*`project_env.yml`* – conda environment with all related packages and modules using the   command line in 3.1
##3.	Execution
3.1.	In order to execute the project,  first make sure you have Anaconda and PyCharm (IDE) installed, then install the project_env.yml: 
```bash
conda env create -f project_env.yml
```
follow the next instructions:  
3.1.1.	Open `PyCharm` in the project root directory
3.1.2.	Go to, File -> Settings -> Python Interpreter -> Add  
![image](https://user-images.githubusercontent.com/104585352/197407096-bc1756c0-4679-46c1-9cb4-51a1994a0630.png)

3.1.3.	Select the Conda Environment that created from the `project_env.yml` file
![image](https://user-images.githubusercontent.com/104585352/197407100-f6783bf3-a862-41c8-9fcf-460fc267382e.png)

3.1.4.	For windows the conda env usually found at
```bash
C:\Anaconda3\envs\<env_name>\python.exe
```
3.2.	Now you are ready to  run the ***`Code/main.py`*** file.  
## 4. Run and Modify Project  
4.1. As described in  section 2 , the ***`Code/main.py`*** file executes and controls all the project aspects, therefore, most of the running, plotting, training, evaluation and data are configuration using this file. So please review  sub-sections `2.1.10` and `2.1.11`  relevant flags and parameters.  
4.2.	**most important running configuration described there**  
4.2.1 `run_over` flag:  
4.2.1.1.	`run_over  = 0` - load all plots from previous results  
4.2.2.2.	`run_over  = 1` - load trained weight and start online evaluation  
4.2.3.3.	`run_over  = 2` - clear all results and train from scratch  
![image](https://user-images.githubusercontent.com/104585352/197407103-35db531c-ea69-47ad-be5f-6def6d682d52.png)

## 5.	Benchmark Results
All results are based on  Model-Based `ViterbiNet` with using  `cost2100` channel model and various DNN architecture.   
5.1.  Coded Bit-Error-Rate Vs SNR Table   
|SNR     |ADNN	   |Sionna	  |SionnaPlus |Transformer|	LSTM	    |ViterbiNet	|ClassicViterbi
|:--------|:------:|:--------:|:---------:|:---------:|:---------:|:---------:|--------------|		
|7|0.041556|0.037167|0.034194|0.042250|0.049028|0.030667|0.022167|
|8|0.029583|0.020000|0.022917|0.025333|0.036583|0.016833|0.011111|
|9|0.018250|0.010528|0.010833|0.014833|0.029167|0.009222|0.003694|
|10|	0.010528|0.005361|0.006333|0.007944|0.015972|0.004139|0.001333|
|11|	0.004056|0.001778|0.001667|0.002333|0.007694|0.001250|0.000194|
|12|	0.003139|0.001000|0.000333|0.000806|0.003694|0.000444|0.000083|
|13|	0.000611|0.000500|0.000111|0.000083|0.000889|0.000167|0.000000|
|14|	0.000389|0.000167|0.000056|0.000000|0.000917|0.000083|0.000000|
|15|	0.000194|0.000278|0.000000|0.000000|0.000806|0.000083|0.000000|  

5.2.   Coded Bit-Error-Rate Vs SNR Chart   
![image](https://user-images.githubusercontent.com/104585352/197407950-e7d44eb6-25bb-4dc8-9a9a-6fcaaab20a03.png)

Enjoy!

