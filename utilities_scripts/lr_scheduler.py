import math 
import functools
import torch

def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return None
    return wrapper

@master_only
def master_only_print(*args):
    """master-only print"""
    print(*args)
    
def get_rank():
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank

class LR_Scheduler(object):
    """Learning Rate Scheduler
    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`
        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0, quiet=False,
                 logger=None):
        self.mode = mode
        self.quiet = quiet
        self.logger = logger
        if not quiet:
            msg = 'Using {} LR scheduler with warm-up epochs of {}!'.format(self.mode, warmup_epochs)
            if self.logger:
                self.logger.info(msg)
            else:
                master_only_print()
        if mode == 'step':
            assert lr_step
        self.base_lr = base_lr
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.total_iters = (num_epochs - warmup_epochs) * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = self.base_lr * 1.0 * T / self.warmup_iters
        elif self.mode == 'cos':
            T = T - self.warmup_iters
            lr = 0.5 * self.base_lr * (1 + math.cos(1.0 * T / self.total_iters * math.pi))
        elif self.mode == 'poly':
            T = T - self.warmup_iters
            lr = self.base_lr * pow((1 - 1.0 * T / self.total_iters), 0.9)
        elif self.mode == 'step':
            lr = self.base_lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplementedError
        if epoch > self.epoch and (epoch == 0 or best_pred > 0.0):
            if not self.quiet:
                msg = '\n=>Epoch %i, learning rate = %.4f, \
                    previous best = %.4f' % (epoch, lr, best_pred)
                if self.logger:
                    self.logger.info(msg)
                else:
                    master_only_print()
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr