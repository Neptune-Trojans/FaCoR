import torch
import os
import random
import numpy as np


def np2tensor(arrays, device='cpu'):
    tensor = torch.from_numpy(arrays).type(torch.float)
    tensor = tensor.to(device)
    return tensor


def mylog(*t, path='log.txt'):
    t = " ".join([str(now) for now in t])
    print(t)
    if os.path.isfile(path) == False:
        f = open(path, 'w+')
    else:
        f = open(path, 'a')
    f.write(t + '\n')
    f.close()


def set_seed(seed):
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def get_device():
    if torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device
