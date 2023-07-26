import os
import os.path as osp
import math
import time
import glob
import abc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from common.timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from common.model import get_model
from dataloader.dataset import MultipleDatasets
dataset_list = ['CrowdPose', 'Human36M', 'MPII', 'MSCOCO', 'MuCo', 'PW3D']
for i in range(len(dataset_list)):
    exec('from dataloader.' + dataset_list[i] + ' import ' + dataset_list[i])
import torch.distributed as dist

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, cfg, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        if dist.get_rank() == 0:
            self.logger = colorlogger(cfg, log_name=log_name)
        else:
            self.logger = None

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    def __init__(self, cfg):
        self.cfg = cfg
        super(Trainer, self).__init__(cfg.log_dir, log_name='train_logs.txt')

    def get_optimizer(self, model):
        base_params = list(map(id, model.module.backbone.parameters()))
        other_params = filter(lambda p: id(p) not in base_params, model.module.parameters())
        optimizer = torch.optim.AdamW([
            {'params': model.module.backbone.parameters(), 'lr': self.cfg.lr_backbone},
            {'params': other_params, },
            {'params': self.awl.parameters(), 'weight_decay': 0}
        ],
        lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(self.cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        if dist.get_rank() == 0:
            self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(self.cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt_path = osp.join(self.cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path, map_location='cpu') 
        start_epoch = ckpt['epoch'] + 1
        info = model.load_state_dict(ckpt['network'], strict=False)
        if cur_epoch != 0:
            self.awl.load_state_dict(ckpt['awl'])
        if dist.get_rank() == 0:
            self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model.cuda(), optimizer

    def set_lr(self, epoch):
        for e in self.cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < self.cfg.lr_dec_epoch[-1]:
            idx = self.cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = self.cfg.lr / (self.cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = self.cfg.lr / (self.cfg.lr_dec_factor ** len(self.cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        if dist.get_rank() == 0:
            self.logger.info("Creating dataset...")
        trainset3d_loader = []
        for i in range(len(self.cfg.trainset_3d)):
            trainset3d_loader.append(eval(self.cfg.trainset_3d[i])(transforms.ToTensor(), "train"))
        trainset2d_loader = []
        for i in range(len(self.cfg.trainset_2d)):
            trainset2d_loader.append(eval(self.cfg.trainset_2d[i])(transforms.ToTensor(), "train"))

        if len(trainset3d_loader) > 0 and len(trainset2d_loader) > 0:
            self.vertex_num = trainset3d_loader[0].vertex_num
            self.joint_num = trainset3d_loader[0].joint_num
            trainset3d_loader = MultipleDatasets(trainset3d_loader, make_same_len=False)
            trainset2d_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
            trainset_loader = MultipleDatasets([trainset3d_loader, trainset2d_loader], make_same_len=True)
        elif len(trainset3d_loader) > 0:
            self.vertex_num = trainset3d_loader[0].vertex_num
            self.joint_num = trainset3d_loader[0].joint_num
            trainset_loader = MultipleDatasets(trainset3d_loader, make_same_len=False)
        elif len(trainset2d_loader) > 0:
            self.vertex_num = trainset2d_loader[0].vertex_num
            self.joint_num = trainset2d_loader[0].joint_num
            trainset_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
        else:
            assert 0, "Both 3D training set and 2D training set have zero length."
            
        self.itr_per_epoch = math.ceil(len(trainset_loader) / self.cfg.num_gpus / self.cfg.train_batch_size)
        if self.cfg.distributed:
            self.sampler = DistributedSampler(trainset_loader)
        else:
            self.sampler = None
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=self.cfg.train_batch_size, 
                shuffle=(self.sampler is None), num_workers=self.cfg.num_thread, pin_memory=True,
                sampler=self.sampler)
                

    def _make_model(self):
        # prepare network
        if dist.get_rank() == 0:
            self.logger.info("Creating graph and optimizer...")
        if not hasattr(self, 'joint_num'):
            self.joint_num = 30
        model = get_model(self.joint_num, 'train', self.cfg)
        awl = AutomaticWeightedLoss(6)

        if self.cfg.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.cuda())
            if self.cfg.is_local:
                model = DDP(model, device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=True)
                awl = DDP(awl.cuda(), device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=True)
            else:
                model = DDP(model, find_unused_parameters=True)
                awl = DDP(awl.cuda(), find_unused_parameters=True)
        else:
            model = DataParallel(model).cuda()
        self.awl = awl
        optimizer = self.get_optimizer(model)
        if self.cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

    def _init_ddp(self):
        torch.cuda.set_device(f'cuda:{self.cfg.local_rank}')
        dist.init_process_group(backend='nccl')
        assert dist.is_initialized(), "Distributed backend not initialized."





class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=21):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True, dtype=torch.float32)
        self.params = nn.Parameter(params)

    def forward(self, loss_dict):
        if not hasattr(self, 'keys'):
            self.keys = sorted(list(loss_dict.keys()))
        for i, key in enumerate(self.keys):
            loss_dict[key] = 0.5 / (self.params[i] ** 2) * loss_dict[key] + torch.log(1 + self.params[i] ** 2)
        return loss_dict