import os
import sys
# main folders
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
CODE_DIR = os.path.join(ROOT_DIR, '')
RESOURCES_DIR = os.path.join(ROOT_DIR, '..\Resources')
RESULTS_DIR = os.path.join(ROOT_DIR, '..\Results')
# subfolders
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
WEIGHTS_DIR = os.path.join(RESULTS_DIR, 'weights')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
COST2100_DIR = os.path.join(RESOURCES_DIR, 'cost2100_channel')
CONFIG_PATH = os.path.join(CODE_DIR, 'configuration.yaml')


# sys.path.append(ROOT_DIR)
# sys.path.append(CODE_DIR)
# sys.path.append(RESOURCES_DIR)
# sys.path.append(RESULTS_DIR)
# sys.path.append(FIGURES_DIR)
# sys.path.append(WEIGHTS_DIR)
# sys.path.append(PLOTS_DIR)
# sys.path.append(COST2100_DIR)
# sys.path.append(CONFIG_PATH)