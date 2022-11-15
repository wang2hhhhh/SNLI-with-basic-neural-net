from tqdm.notebook import tqdm_notebook
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import *
from model import *
from data_utils import *
import argparse
import json


args.max_sentence_length
args.data_folder


def _parse_args():

    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--out_dir', type=str, help='path to output folder for saving the trained model')

    parser.add_argument('--train_path', type=str, help='path to train folder that contain the tokenized training set')

    parser.add_argument('--dev_path', type=str, help='path to train folder that contain the tokenized validation set')

    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')

    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')

    parser.add_argument('--hid_dim', type=int, default=128, help='embedding hidden layer size')

    parser.add_argument('--emb_dim', type=int, default=256, help='linear hidden layer size')

    parser.add_argument('--reduce', type=str, default= "hadamard", help=' choose from "sum", "hadamard", "concat" ')

    parser.add_argument('--batch_size', type=int, default=16, help='training batch size; 16 by default ')

    parser.add_argument('--max_sentence_length', type=int, default=30, help='maximum length of input sentences')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    #get train, validation dataset for training
    train_dataloader, validation_dataloader, vocab_size = batchify(args)

    #init model
    model = LogisticsModelPytorch(vocab_size, args.emb_dim, args.hid_dim, 3)

    # Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # record losses
    loss_dict = {

        train_losses : [],
        val_losses : [] ,
        train_accs : [],
        val_accs : [],
        train_losses_batch : [],
        val_losses_batch : [],
        train_accs_batch : [],
        val_accs_batch : []

    }

    # train_model
    loss_dict, model = train(
                        model, train_dataloader, validation_dataloader,
                        args, criterion, optimizer, loss_dict)

    #save model
    torch.save(model, args.out_dir)

    #save loss
    with open(os.path.join(args.out_dir, 'training_stats'), "w") as outfile:
        json.dump(loss_dict, outfile) 


