# Copyright (c) Facebook, Inc. and its affiliates.
import math
import datetime
import torch.distributed as dist
import logging
import os
import torch
import psutil


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def save_checkpoint(state, savedir, itr, last_checkpoints=None, num_checkpoints=None):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    filename = os.path.join(savedir, 'checkpt-%08d.pth' % itr)
    torch.save(state, filename)

    if last_checkpoints is not None and num_checkpoints is not None:
        last_checkpoints.append(itr)
        if len(last_checkpoints) > num_checkpoints:
            rm_itr = last_checkpoints.pop(0)
            old_checkpt = os.path.join(savedir, 'checkpt-%08d.pth' % rm_itr)
            if os.path.exists(old_checkpt):
                os.remove(old_checkpt)


def find_latest_checkpoint(savedir):
    import glob
    import re

    checkpt_files = glob.glob(os.path.join(savedir, 'checkpt-[0-9]*.pth'))

    if not checkpt_files:
        return None

    def extract_itr(f):
        s = re.findall('(\d+).pth$', f)
        return int(s[0]) if s else -1

    latest_itr = max(checkpt_files, key=extract_itr)
    return latest_itr


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


class ExponentialMovingAverage(object):

    def __init__(self, module, decay=0.999):
        """Initializes the model when .apply() is called the first time.
        This is to take into account data-dependent initialization that occurs in the first iteration."""
        self.decay = decay
        self.module_params = {n: p for (n, p) in module.named_parameters()}
        self.ema_params = {n: p.data.clone() for (n, p) in module.named_parameters()}
        self.nparams = sum(p.numel() for (_, p) in self.ema_params.items())

    def apply(self, decay=None):
        decay = decay or self.decay
        with torch.no_grad():
            for name, param in self.module_params.items():
                self.ema_params[name] -= (1 - decay) * (self.ema_params[name] - param.data)

    def set(self, named_params):
        with torch.no_grad():
            for name, param in named_params.items():
                self.ema_params[name].copy_(param)

    def replace_with_ema(self):
        for name, param in self.module_params.items():
            param.data.copy_(self.ema_params[name])

    def swap(self):
        for name, param in self.module_params.items():
            tmp = self.ema_params[name].clone()
            self.ema_params[name].copy_(param.data)
            param.data.copy_(tmp)

    def __repr__(self):
        return (
            '{}(decay={}, module={}, nparams={})'.format(
                self.__class__.__name__, self.decay, self.module.__class__.__name__, self.nparams
            )
        )


def get_msle(prediction, event_popularity):
    true = torch.unsqueeze(event_popularity, 1)
    true = true + 1
    prediction = prediction + 1

    true = torch.log2(true)
    prediction = torch.log2(prediction)

    diff = prediction - true
    msle = diff * diff

    return msle


def get_msape(prediction, event_popularity):
    true = torch.unsqueeze(event_popularity, 1)
    true = true + 2
    prediction = prediction + 2

    true = torch.log2(true)
    prediction = torch.log2(prediction)

    diff = prediction - true
    msape = torch.abs(diff) / true

    return msape


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))


def cleanup():
    dist.destroy_process_group()


def memory_usage_psutil():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def cosine_decay(learning_rate, global_step, decay_steps, alpha=0.0):
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return learning_rate * decayed


def learning_rate_schedule(global_step, warmup_steps, base_learning_rate, train_steps):
    warmup_steps = int(round(warmup_steps))
    scaled_lr = base_learning_rate
    if warmup_steps:
        learning_rate = global_step / warmup_steps * scaled_lr
    else:
        learning_rate = scaled_lr

    if global_step < warmup_steps:
        learning_rate = learning_rate
    else:
        learning_rate = cosine_decay(scaled_lr, global_step - warmup_steps, train_steps - warmup_steps)
    return learning_rate


def set_learning_rate(optimizer, lr):
    for i, group in enumerate(optimizer.param_groups):
        group['lr'] = lr


def cast(tensor, device):
    return tensor.float().to(device) if torch.is_tensor(tensor) else None


def to_numpy(x):
    if torch.is_tensor(x):
        return x.cpu().detach().numpy()
    return [to_numpy(x_i) for x_i in x]


def get_t0_t1(t1):
    if t1:
        return torch.tensor([0.0]), torch.tensor([1.0])
    else:
        return torch.tensor([0.0]), None
