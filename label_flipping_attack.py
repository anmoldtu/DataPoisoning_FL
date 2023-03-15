from federated_learning.utils import replace_0_with_2
from federated_learning.utils import replace_5_with_3
from federated_learning.utils import replace_1_with_9
from federated_learning.utils import replace_4_with_6
from federated_learning.utils import replace_1_with_3
from federated_learning.utils import replace_6_with_0
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--NUM_EXP')
    parser.add_argument('--START_EXP_IDX')
    parser.add_argument('--NET')
    parser.add_argument('--LR')
    parser.add_argument('--NOISE')
    parser.add_argument('--CLIP')
    args = parser.parse_args()
    
    START_EXP_IDX = int(args.START_EXP_IDX)
    NUM_EXP = int(args.NUM_EXP)
    NET = args.NET
    LR = args.LR
    NUM_POISONED_WORKERS = 0
    REPLACEMENT_METHOD = replace_1_with_9
    KWARGS = {
        "NUM_WORKERS_PER_ROUND" : 5,
        "NOISE" : float(args.NOISE),
        "CLIP" : float(args.CLIP)
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id, NET, LR)
