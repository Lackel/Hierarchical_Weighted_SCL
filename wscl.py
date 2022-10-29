from init_parameter import init_model
from data import Data
from model import ModelManager
from transformers import logging
import os

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.set_verbosity_error()
    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)

    manager = ModelManager(args, data)
    
    print('Training begin...')
    manager.train()
    print('Training finished!')

    print('Evaluation begin...')
    metric = manager.test()
    print('Evaluation finished!')
    print(metric)
