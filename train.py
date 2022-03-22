import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
import warnings
import argparse
import logging
import sys
from tqdm import tqdm
import numpy as np
from loss import dice_bce_loss
from utils.metrics import Evaluator
from datasets import build_train_dataset, build_val_dataset
from utils.lr_scheduler import LR_Scheduler
from models import build_model
from utils.calculate_weights import calculate_weigths_labels
from loss import Losses
from models.utils import resize
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class Trainer(object):
    def __init__(self, args):
        self.args = args

        train_dataset = build_train_dataset(args)
        val_dataset = build_val_dataset(args)
        if args.distributed:
            self.train_sampler = DistributedSampler(train_dataset)
            self.val_sampler = DistributedSampler(val_dataset)
        else:
            self.train_sampler = torch.utils.data.RandomSampler(train_dataset)
            self.val_sampler = torch.utils.data.SequentialSampler(val_dataset)

        self.data_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=self.train_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers)

        self.val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=self.val_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers)

        self.no_optim = 0
        self.best_pred = 0.
        self.total_epoch = args.epochs
        self.train_epoch_best_loss = 100.

        # Define Network
        self.model = build_model(args.model_name, num_classes=args.num_classes, sync_bn=args.sync_bn)
        if args.cuda:
            self.model = self.model.cuda()
            if args.distributed:
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, 
                            device_ids=[args.local_rank], find_unused_parameters=True)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            self.model.load_state_dict(checkpoint)
            logging.info("=> loaded checkpoint '{}' ".format(args.resume))

        no_decay_list = ['projection', 'norm', 'relative_position_bias_table', 'bn']
        params_no_decay = [p for n, p in list(self.model.named_parameters()) if any(nd in n for nd in no_decay_list)]
        params_with_decay = [p for n, p in list(self.model.named_parameters()) if not any(nd in n for nd in no_decay_list)]
        train_params = [{'params': params_no_decay, 'lr': args.lr, 'weight_decay': 0},
                        {'params': params_with_decay, 'lr': args.lr, 'weight_decay': args.weight_decay}]

        self.optimizer = torch.optim.AdamW(params=train_params, betas=(0.9, 0.999), lr=args.lr)

        if args.use_balanced_weights:
            classes_weights_path = os.path.join(args.working_dir + 'output/', args.model_name + '_' + args.dataset_name +'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(self.data_loader, args.num_classes, 
                                            args.working_dir + 'output/', args.model_name)
            weight[1] = weight[1] * 5 # more weights for road class
            print('Weights for classes: ' + str(weight))
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        # self.criterion = torch.nn.CrossEntropyLoss(weight=weight).cuda()
        self.criterion = Losses(loss_decode=args.loss_fn)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.data_loader), 
                                warmup_epochs=args.warmup_epochs, lr_1x_param_list=[0, 1], lr_2x_param_list=[])

        self.evaluator = Evaluator(2)
        # self.scaler = GradScaler()
        self.iters_to_accumulate = args.iters_to_accumulate

    def training(self, epoch):
        self.model.train()
        if self.args.distributed:
            self.train_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            logging.info('\n=>Epoches %i, learning rate = %.7f, \
                    previous best = %.4f' % (epoch, self.scheduler.get_current_lr(), self.best_pred))
        train_epoch_loss = 0
        tbar = tqdm(self.data_loader)
        for i, data in enumerate(tbar):
            img, mask = data['img'], data['label']

            pred = self.model.forward(img.cuda())
            train_loss = self.criterion.build_losses(pred, mask.cuda().long())
            train_loss = self.criterion.count_total_loss(train_loss) / self.iters_to_accumulate
            
            train_loss.backward()
            if (i + 1) % self.iters_to_accumulate == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.scheduler(self.optimizer, i, epoch - 1)
            
            tbar.set_description('Train loss: %.4f ' % train_loss * self.iters_to_accumulate)
            train_epoch_loss += train_loss * self.iters_to_accumulate
        train_epoch_loss /= len(self.data_loader)
        torch.cuda.synchronize()
        train_epoch_loss = self.reduce_tensor(torch.tensor(train_epoch_loss)) / dist.get_world_size()
        if args.local_rank == 0:
            logging.info('Train: epoch:' + str(epoch) + ', train_loss: %.5f' % train_epoch_loss.item())
        return train_epoch_loss.item()


    def validation(self, epoch): 
        self.evaluator.reset()
        if self.args.distributed:
            self.val_sampler.set_epoch(epoch)
        with torch.no_grad():
            self.model.eval()
            val_epoch_loss = 0.0
            tbar = tqdm(self.val_data_loader)
            for data in tbar:
                val_img, val_mask = data['img'], data['label']
                val_img, val_mask = val_img.cuda(), val_mask.cuda()
                pred = self.model.forward(val_img)
                # pred = pred.squeeze()
                val_loss = self.criterion.build_losses(pred, val_mask.cuda().long())
                val_loss = self.criterion.count_total_loss(val_loss)
                pred = resize(
                    input=pred,
                    size=val_mask.shape[1:],
                    mode='bilinear',
                    align_corners=False)
                pred = F.softmax(pred, dim=1)
                pred = pred.data.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                val_mask = val_mask.data.cpu().numpy()
                tbar.set_description('Val loss: %.4f ' % val_loss)
                self.evaluator.add_batch(val_mask, pred)
                val_epoch_loss += val_loss

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()
        Precision = self.evaluator.Pixel_Precision()
        Recall = self.evaluator.Pixel_Recall()
        F1 = self.evaluator.Pixel_F1()

        torch.cuda.synchronize()
        IoU = self.reduce_tensor(torch.tensor(IoU).cuda()) / dist.get_world_size()
        if args.local_rank == 0:
            logging.info("Validation: Val_loss:%.7f, Acc:%.7f, Acc_class:%.7f, mIoU:%.7f, IoU:%.7f, Precision:%.7f, Recall:%.7f, F1:%.7f"
                % (val_epoch_loss / len(self.val_data_loader), Acc, Acc_class, mIoU, IoU.item(), Precision, Recall, F1))
        return IoU

    def update_state(self, epoch, train_epoch_loss, IoU):
        if train_epoch_loss >= self.train_epoch_best_loss and IoU < self.best_pred:
            self.no_optim += 1
        else:
            self.no_optim = 0
            if IoU > self.best_pred:
                torch.save(self.model.state_dict(), args.working_dir + 'weights/' + 
                                   self.args.model_name + '_' + self.args.dataset_name + '.th')
            self.train_epoch_best_loss = train_epoch_loss if self. \
                train_epoch_best_loss > train_epoch_loss else self.train_epoch_best_loss
            self.best_pred = IoU if self.best_pred < IoU else self.best_pred
            
        # if self.no_optim > 6:
        #     logging.info('early stop at %d epoch\n' % epoch)
        #     return False
        # if self.no_optim > 3:
        #     if self.scheduler.get_current_lr() < 5e-7:
        #         return False
        #     self.model.load_state_dict(torch.load('Project/D-LinkNet/weights/' + self.args.model_name + '.th'))
        #     self.scheduler.update_current_lr(0.5, factor=True)
        return True


    def reduce_tensor(self, tensor):
        # sum the tensor data across all machines
        dist.all_reduce(tensor, op=dist.reduce_op.SUM)
        return tensor


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="PyTorch RoadNet Training")
    parser.add_argument('--base-size', type=int, default=1300,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=1024,
                        help='crop image size')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='number of classes in dataset')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size per gpu for training(default: 4)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos', 'cond_step'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--loss-fn', type=list, default=[dict(type='CrossEntropyLoss', loss_weight=1.0),
                        dict(type='DiceLoss', loss_weight=1.0)])
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='lr linear warmup epochs')
    parser.add_argument('--iters-to-accumulate', type=int, default=1,
                        help='iterations accumulated to update weight once')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync-bn in model')
    parser.add_argument('--model-name', type=str, default='Swin_UperNet',
                        help='model name for training')
    parser.add_argument('--dataset-name', type=str, default='SpaceNet',
                        help='dataset name used in training process')
    parser.add_argument('--dataset-basedir', type=str, default='/root/share/SpaceNet_dataset/',
                        help='base dir of dataset corresponding to the dataset')
    parser.add_argument('--gpu-nums', type=int, default=1,
                        help='how many gpus to train (default: 1)')
    parser.add_argument('--no-cuda', type=bool, default=False, 
                        help='disables CUDA training')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--working_dir', type=str, default='/root/SwinCNN_UperNet/',
                        help='output directory')
    parser.add_argument('--use-balanced-weights', type=bool, default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist_init_method', type=str, default='env://')
    args = parser.parse_args()

    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.gpu_nums = int(os.environ['WORLD_SIZE'])

    if args.distributed:
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method=args.dist_init_method, rank=args.local_rank)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logging.basicConfig(filename=args.working_dir + "logs/" + args.model_name + '_' + args.dataset_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    if args.local_rank == 0:
        logging.info(args)
        
    trainer = Trainer(args)
    trainer.scheduler(trainer.optimizer, 0, 0)
    for epoch in range(1, args.epochs + 1):
        train_epoch_loss = trainer.training(epoch)
        if epoch % 1 == 0:
            IoU = trainer.validation(epoch)
        if args.local_rank == 0:
            state = trainer.update_state(epoch, train_epoch_loss, IoU)
        # if not state:
        #     break
    logging.info('Finish!')


