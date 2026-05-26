import torch
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import Normalizer


def delete_item_in_history(tensor, indices, h=50):
    """
    Remove items in the indices from the tensor
    """
    return tensor[~tensor.unsqueeze(1).eq(indices[-h:]).any(1)]


def set_seed(seed, cuda=False):
    """
    Set the random seed
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    if cuda:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def assert_no_grad(variable):
    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )


def str2bool(v):
    return v.lower() in ('true')


def get_end_index(seq, pad=0):
    """
    Obtain the end position of sequence, given post-padding.
    """
    pos = np.where(seq == pad)[0]
    if len(pos) == 0:
        return len(seq) - 1
    else:
        return pos[0] - 1


def get_start_index(seq, pad=0):
    """
    Obtain the start position of sequence, given pre-padding.
    The sequence cannot be all padding.
    """
    return np.where(seq != pad)[0][0]


def get_item_index(seq, item):
    """
    Obtain the position of an item in a sequence.
    """
    pos = np.where(seq == item)[0]
    if len(pos) == 0:
        return -1
    else:
        return pos[0]


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, save_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}'
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        # torch.save(
        #     model, save_path + "/" +
        #     "checkpoint_{}_{:.6f}.pth.tar".format(epoch, val_loss))
        if model != None:
            torch.save(model, save_path)
        self.val_loss_min = val_loss





