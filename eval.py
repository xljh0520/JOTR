import argparse
import os
import torch
import torch.distributed as dist
import sys
from config import cfg
from base import Trainer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader.PW3D import PW3D
from dataloader.CMU_Panotic import CMU_Panotic
import sys
import numpy as np
import random
sys.path.insert(0, os.path.join(cfg.root_dir, 'common'))
from utils.dir import make_folder
    
def setup_seed(seed=42):
    seed += dist.get_rank()
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = False  
    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.enabled = True

best_dict = {
    '3dpw': {
        'best_MPJPE': 1e10,
    },
    '3dpw-crowd':{
        'best_MPJPE': 1e10,
    },
    '3dpw-pc':{
        'best_MPJPE': 1e10,
    },
    '3dpw-oc':{
        'best_MPJPE': 1e10,
    },
}
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='experiment configure file name')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--exp_id', type=str, default='debug', help='experiment configure file name')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--resume_ckpt', type=str, default='', help='for resuming train')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.distributed:
        torch.cuda.set_device(args.local_rank) 
        dist.init_process_group(backend='nccl', init_method='env://')
        assert dist.is_initialized(), "distributed is not initialized"
    if dist.get_rank() == 0:
        make_folder(cfg.model_dir)
        make_folder(cfg.vis_dir)
        make_folder(cfg.log_dir)
        make_folder(cfg.result_dir)
        dirs = [cfg.model_dir, cfg.vis_dir, cfg.log_dir, cfg.result_dir]
    else:
        dirs = [None, None, None, None]
    dist.broadcast_object_list(dirs, src=0)
    cfg.model_dir, cfg.vis_dir, cfg.log_dir, cfg.result_dir = dirs
    setup_seed()
    if dist.get_rank() == 0:
        cfg.set_args(args.continue_train, resume_ckpt=args.resume_ckpt)
    if args.cfg:
        yml_cfg = cfg.update(args)
    trainer = Trainer(cfg)
    trainer._make_model()
    test_dataset_dict = {}
    for dataset_name in best_dict:
        if '3dpw' in dataset_name:
            testset_loader = PW3D(transforms.ToTensor(), data_name=dataset_name)
        else:
            testset_loader = CMU_Panotic()
        if cfg.distributed:
            testset_sampler = torch.utils.data.distributed.DistributedSampler(testset_loader)
        else:
            testset_sampler = None
        test_batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.test_batch_size, 
                    shuffle=False, num_workers=cfg.num_thread, pin_memory=True,
                    sampler=testset_sampler
                    )
        test_dataset_dict[dataset_name] = {
            'loader': test_batch_generator,
            'dataset': testset_loader
        }
    for data_name in best_dict.keys():
        ckpt_path = os.path.join('./checkpoint', '{}_best_ckpt.pth.tar'.format(data_name))
        ckpt = torch.load(ckpt_path, map_location='cpu')
        trainer.model.load_state_dict(ckpt)
        trainer.model.eval()
        # eval(0, trainer, data_name, test_dataset_dict[data_name]['dataset'], test_dataset_dict[data_name]['loader'])

def eval(epoch, trainer, dataset_name, testset_loader, test_batch_generator):
    trainer.model.eval()
    eval_result = {}
    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(test_batch_generator)):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                out = trainer.model(inputs, targets, meta_info, 'test')
        out = {k: v.cpu().numpy() for k,v in out.items()}
        key = list(out.keys())[0]
        batch_size = out[key].shape[0]
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)] # batch_size * dict
        if not dist.is_initialized():
            cur_eval_result = testset_loader.evaluate(out, cur_sample_idx) # dict of list
            for k,v in cur_eval_result.items():
                if k in eval_result: 
                    eval_result[k] += v
                else: 
                    eval_result[k] = v
            cur_sample_idx += len(out)
        else:
            index_list = meta_info['idx'].flatten().long().tolist()
            cur_eval_result = testset_loader.random_idx_eval(out, index_list)
            for k,v in cur_eval_result.items():
                if k in eval_result: 
                    eval_result[k] += v
                else: 
                    eval_result[k] = v
    mpjpe = torch.tensor(np.mean(eval_result['mpjpe'])).float().cuda().flatten()
    pa_mpjpe = torch.tensor(np.mean(eval_result['pa_mpjpe'])).float().cuda().flatten()
    mpvpe = torch.tensor(np.mean(eval_result['mpvpe'])).float().cuda().flatten()
    samples = torch.tensor(len(eval_result['mpjpe'])).float().cuda().flatten()
    dist.barrier()
    gather_list = [torch.zeros_like(mpjpe) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, mpjpe)
    mpjpe_pre_rank = torch.stack(gather_list).flatten()
    dist.all_gather(gather_list, pa_mpjpe)
    pa_mpjpe_pre_rank = torch.stack(gather_list).flatten()
    dist.all_gather(gather_list, mpvpe)
    mpvpe_pre_rank = torch.stack(gather_list).flatten()
    dist.all_gather(gather_list, samples)
    samples_pre_rank = torch.stack(gather_list).flatten()
    
    all_samples = samples_pre_rank.sum()
    all_mpjpe = mpjpe_pre_rank * samples_pre_rank
    all_pa_mpjpe = pa_mpjpe_pre_rank * samples_pre_rank
    all_mpvpe = mpvpe_pre_rank * samples_pre_rank

    mean_mpjpe = all_mpjpe.sum() / all_samples
    mean_pa_mpjpe = all_pa_mpjpe.sum() / all_samples
    mean_mpvpe = all_mpvpe.sum() / all_samples
    result_dict = {
        'mpjpe': mean_mpjpe.item(),
        'pa_mpjpe': mean_pa_mpjpe.item(),
        'mpvpe': mean_mpvpe.item(),
    }
        
    if dist.get_rank() == 0:
        print('{} {}'.format(dataset_name, epoch))
        for k,v in result_dict.items():
            print(f'{k}: {v:.2f}')
        
        message = [f'{k}: {v:.2f}' for k, v in result_dict.items()]
        # message = ' '.join(message)
        trainer.logger.info('{} '.format(dataset_name) + ' '.join(message))         
        
    dist.barrier()

if __name__ == "__main__":
    main()
