from argparse import ArgumentParser

def init_model():
    parser = ArgumentParser()
    # data
    parser.add_argument("--data_dir", default='data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--dataset", default=None, type=str, required=True, 
                        help="The name of the dataset to train selected.")
    # model
    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id.")
    
    parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")

    parser.add_argument("--model_name", default="bert-base-uncased", type=str, help="The path or name of the pre-trained BERT model.")
    
    parser.add_argument("--freeze_bert_parameters", action="store_true", help="Freeze the last parameters of BERT.")

    # hyperparameters
    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")

    parser.add_argument("--layer_num", default=8, type=int, help="The index of the shallow layer.")
    
    parser.add_argument("--warmup_proportion", default=0.1, type=float)

    parser.add_argument("--momentum_factor", default=0.9, type=float, help="The weighting factor of the momentum BERT.")

    parser.add_argument("--alpha_m", default=1.0, type=float, help="The weighting factor for momentum negative keys.")

    parser.add_argument("--alpha_diff", default=1.4, type=float, help="The weighting factor for momentum negative key with different coarse labels as the query.")

    parser.add_argument("--alpha_same", default=1.0, type=float, help="The weighting factor for negative keys with the same coarse labels as the query.")

    parser.add_argument("--gamma1", default=0.001, type=float, help="The weighting factor for cross entropy loss at the shallow layer.")

    parser.add_argument("--gamma2", default=0.008, type=float, help="The weighting factor for the weighted self-contrastive loss.")

    # training and testing
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size for training.")
    
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")

    parser.add_argument("--num_train_epochs", default=20, type=float,
                        help="The training epochs.")

    parser.add_argument("--lr_pre", default=5e-5, type=float,
                        help="The learning rate for pre-training.")
                    
    parser.add_argument("--temperature", default=0.1, type=float,
                        help="The temperature for dot product.")
    
    parser.add_argument("--lr", default=5e-5, type=float,
                        help="The learning rate for training.")  
    
    
    return parser
