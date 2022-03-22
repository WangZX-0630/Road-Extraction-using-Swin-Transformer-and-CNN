import math
import logging
import sys


class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0, min_lr=1e-6, lr_1x_param_list=[0], lr_2x_param_list=[]):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.min_lr=min_lr
        self.current_lr = base_lr
        self.no_optim = 0
        self.best_pred = 0.0
        self.current_epoch_loss = 999.9
        self.lr_1x_param_list = lr_1x_param_list
        self.lr_2x_param_list = lr_2x_param_list
        
        
    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 1.0)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        elif self.mode == 'cond_step':
            lr = self.current_lr
        else:
            raise NotImplementedError
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = self.lr * 1.0 * T / self.warmup_iters + self.min_lr
        if epoch > self.epoch:
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)
        self.current_lr = lr


    def update_cond_step_lr(self, epoch_loss, best_pred):
        if self.mode == 'cond_step':
            if epoch_loss > self.current_epoch_loss or best_pred < self.best_pred:
                self.no_optim += 1
            else:
                self.current_epoch_loss = epoch_loss
                self.best_pred = best_pred
                self.no_optim = 0

            if self.no_optim > 3:
                self.current_lr = self.current_lr * 0.5
                self.no_optim = 0


    def update_current_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.current_lr * new_lr
        logging.info('update learning rate: %f -> %f' % (self.current_lr, new_lr))
        self.current_lr = new_lr


    def get_current_lr(self):
        return self.current_lr

    def _adjust_learning_rate(self, optimizer, lr):
        if len(self.lr_2x_param_list) == 0:
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr
        else:
            for i in self.lr_1x_param_list:
                optimizer.param_groups[i]['lr'] = lr
            for i in self.lr_2x_param_list:
                optimizer.param_groups[i]['lr'] = lr * 2
