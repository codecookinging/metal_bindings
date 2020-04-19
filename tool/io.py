from __future__ import print_function, division
import os
import errno
import shutil
import torch

def save_checkpoint(state, folder, filename):
    """Save state dictionary into disk.
    Save state dictionary in the location folder/filename
    """

    try:
        os.makedirs(folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

    torch.save(state, os.path.join(folder, filename))

def load_checkpoint(folder, filename):
    """Load checkpoint dictionary into memory from the location folder/filename
    Args:
        folder (str): Name of the folder that contains the dictionary.
        filename (str): Name of the dictionary.
    """
    filename = os.path.join(folder, filename)
    # if os.path.isfile(filename):
    try:
        checkpoint = torch.load(filename)
    except IOError:
        print("'{}' was not found, please check that '{}' exists".format(
            filename, filename))
        raise
    return checkpoint