import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import warnings

warnings.filterwarnings("ignore")

import time, cv2, torch, wandb
import torch.distributed as dist
from config.diffconfig import DiffusionConfig, get_model_conf
from config.dataconfig import Config as DataConfig
from tensorfn import load_config as DiffConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
from tensorfn.optim import lr_scheduler
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import data as deepfashion_data
from model import UNet
from torchvision.utils import make_grid

from config_reid_model import cfg_reid_model  # reid model config
from reid_model import make_model


def init_distributed():
    import os
    import torch
    import torch.distributed as dist

    # Determine whether it is a distributed startup
    if "RANK" not in os.environ:
        print("==> Running in single GPU mode")
        return 0, False  # local_rank, is_distributed

    # environment variables provided by torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

    dist.barrier()

    setup_for_distributed(rank == 0)

    print(f"==> Distributed init done | rank={rank}, local_rank={local_rank}")

    return local_rank, True


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_main_process():  # whether the current process is the main process
    try:
        if dist.get_rank()==0:
            return True
        else:
            return False
    except:
        return True

def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        # ema = decay * ema + (1-decay) * current
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)




def train(conf, loader, val_loader, model, ema, diffusion, betas, optimizer, scheduler, guidance_prob, cond_scale, device, wandb, use_id_loss=False, id_loss_weight=1.0, id_loss_k=1.0, begin_id_loss=50):

    import time

    i = 0  # iteration count

    # recording loss
    loss_list = []
    loss_mse_list = []
    loss_vb_list = []
    loss_id_list = []
 
    for epoch in range(600):  # 300 600

        if is_main_process: print ('#Epoch - '+str(epoch))

        start_time = time.time()

        for batch in tqdm(loader):

            i = i + 1

            # ======================
            # constructing input
            # ======================

            # img = torch.cat([batch['source_image'], batch['target_image']], 0)                  # xs
            # target_img = torch.cat([batch['target_image'], batch['source_image']], 0)           # y0
            # target_pose = torch.cat([batch['target_skeleton'], batch['source_skeleton']], 0)    # xp

            # negative_img = torch.cat([batch['negative_target_image'], batch['negative_source_image']], 0)  # for triplet id loss

            img = batch["source_image"]              # xs
            target_img = batch["target_image"]       # y0
            target_pose = batch["target_skeleton"]   # xp
            ''' mask has already been concatenated with the pose in dataloader '''
            negative_img = batch["negative_target_image"]   # for triplet id loss

            img = img.to(device)
            target_img = target_img.to(device)
            target_pose = target_pose.to(device)
            negative_img = negative_img.to(device)

            # sampling diffusion time step
            time_t = torch.randint(
                0,
                conf.diffusion.beta_schedule["n_timestep"],
                (img.shape[0],),
                device=device,
            )

            # ======================
            # calculate the loss
            # ======================
            enable_id_loss = use_id_loss and (epoch >= begin_id_loss)
            loss_dict = diffusion.training_losses(
                model,
                x_start=target_img,
                t=time_t,
                cond_input=[img, target_pose],
                negative_img=negative_img,
                prob=1 - guidance_prob,
                reid_model=reid_model if enable_id_loss else None,
                use_id_loss=enable_id_loss,
                id_loss_weight=id_loss_weight,
                id_loss_k=id_loss_k,
            )

            loss = loss_dict['loss'].mean()
            loss_mse = loss_dict['mse'].mean()
            loss_vb = loss_dict['vb'].mean()
            loss_id = loss_dict.get("triplet_id", torch.zeros_like(loss)).mean()

            # print('loss_id:', loss_id)
            # print('loss_mse:', loss_mse)
            # print('loss_vb:', loss_vb)
        

            # ======================
            # backward
            # ======================
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            scheduler.step()

            loss = loss_dict['loss'].mean()

            loss_list.append(loss.detach().item())
            loss_mse_list.append(loss_mse.detach().item())
            loss_vb_list.append(loss_vb.detach().item())
            loss_id_list.append(loss_id.detach().item())

            model_to_use = model.module if hasattr(model, "module") else model
            accumulate(
                ema, model_to_use, 0 if i < conf.training.scheduler.warmup else 0.9999
            )

            # ======================
            # wandb logs
            # ======================
            if i%args.save_wandb_logs_every_iters == 0 and is_main_process():

                wandb.log({'loss':(sum(loss_list)/len(loss_list)), 
                            'loss_vb':(sum(loss_vb_list)/len(loss_vb_list)), 
                            'loss_mse':(sum(loss_mse_list)/len(loss_mse_list)), 
                            'loss_id':(sum(loss_id_list)/len(loss_id_list)), 
                            'epoch':epoch,'steps':i})
                loss_list = []
                loss_mse_list = []
                loss_vb_list = []
                loss_id_list = []


            # ======================
            # save checkpoint
            # ======================
            if i%args.save_checkpoints_every_iters == 0 and is_main_process():

                if conf.distributed:
                    model_module = model.module

                else:
                    model_module = model

                torch.save(
                    {
                        "model": model_module.state_dict(),
                        "ema": ema.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "conf": conf,
                    },
                    conf.training.ckpt_path + f"/model_{str(i).zfill(6)}.pt"
                )


        # ======================
        # epoch end
        # ======================
        if is_main_process():

            print ('Epoch Time '+str(int(time.time()-start_time))+' secs')
            print ('Model Saved Successfully for #epoch '+str(epoch)+' #steps '+str(i))

            if conf.distributed:
                model_module = model.module

            else:
                model_module = model

            torch.save(
                {
                    "model": model_module.state_dict(),
                    "ema": ema.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "conf": conf,
                },
                conf.training.ckpt_path + '/last.pt'
               
            )


        # ======================
        # generate samples (for verification visualization)
        # ======================
        if (epoch)%args.save_wandb_images_every_epochs==0:

            print ('Generating samples at epoch number ' + str(epoch))

            val_batch = next(val_loader)

            val_img = val_batch['source_image'].cuda()
            val_pose = val_batch['target_skeleton'].cuda()

            with torch.no_grad():

                if args.sample_algorithm == 'ddpm':
                    print ('Sampling algorithm used: DDPM')
                    samples = diffusion.p_sample_loop(ema, x_cond = [val_img, val_pose], progress = True, cond_scale = cond_scale)
                elif args.sample_algorithm == 'ddim':
                    print ('Sampling algorithm used: DDIM')
                    nsteps = 50
                    noise = torch.randn(val_img.shape).cuda()
                    seq = range(0, 1000, 1000//nsteps)
                    xs, x0_preds = ddim_steps(noise, seq, ema, betas.cuda(), [val_img, val_pose])
                    samples = xs[-1].cuda()

            grid = torch.cat([val_img, val_pose[:,:3], samples], -1)
            
            if is_distributed:
                gathered_samples = [torch.zeros_like(grid) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, grid) 
                samples = torch.cat(gathered_samples, dim=0)
            else:
                samples = grid

            samples = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
            
            if is_main_process():
                '''
                wandb.log({'samples':wandb.Image(torch.cat(gathered_samples, -2))})
                '''
                wandb.log({'samples': wandb.Image(samples)})


def main(settings, EXP_NAME, local_rank=0, is_distributed=False, use_id_loss=False, id_loss_weight=1.0, id_loss_k=1.0, begin_id_loss=50):

    [args, DiffConf, DataConf] = settings

    # init wandb
    if is_main_process(): wandb.init(project="person-synthesis", name = EXP_NAME,  settings = wandb.Settings(code_dir="."))

    if DiffConf.ckpt is not None: 
        DiffConf.training.scheduler.warmup = 0
    

    # DiffConf.distributed = True
    DiffConf.distributed = is_distributed
    
    DataConf.data.train.batch_size = args.batch_size//2  #src -> tgt , tgt -> src
    
    # load dataset
    # val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(DataConf.data, labels_required = True, distributed = True)
    val_dataset, train_dataset = deepfashion_data.get_train_val_dataloader(DataConf.data, labels_required = True, distributed = is_distributed)
    
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    val_dataset = iter(cycle(val_dataset))

    # ======================
    # build model
    # ======================
    model = get_model_conf().make_model()
    model = model.to(args.device)
    ema = get_model_conf().make_model()
    ema = ema.to(args.device)

    if DiffConf.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True
        )

    optimizer = DiffConf.training.optimizer.make(model.parameters())
    scheduler = DiffConf.training.scheduler.make(optimizer)

    # ======================
    # load checkpoint
    # ======================
    if DiffConf.ckpt is not None:
        ckpt = torch.load(DiffConf.ckpt, map_location=lambda storage, loc: storage)

        if DiffConf.distributed:
            model.module.load_state_dict(ckpt["model"])

        else:
            model.load_state_dict(ckpt["model"])

        ema.load_state_dict(ckpt["ema"])
        scheduler.load_state_dict(ckpt["scheduler"])

        if is_main_process():  print ('model loaded successfully')

    # ======================
    # diffusion process
    # ======================
    betas = DiffConf.diffusion.beta_schedule.make()
    diffusion = create_gaussian_diffusion(betas, predict_xstart = False)


    train(
        DiffConf, train_dataset, val_dataset, model, ema, diffusion, betas, optimizer, scheduler, args.guidance_prob, args.cond_scale, args.device, wandb, use_id_loss=use_id_loss, id_loss_weight=id_loss_weight, id_loss_k=id_loss_k, begin_id_loss=begin_id_loss
    )



if __name__ == "__main__":

    local_rank, is_distributed = init_distributed()

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import argparse

    # ======================
    # config
    # ======================
    parser = argparse.ArgumentParser(description='help')

    parser.add_argument('--exp_name', type=str, default='pidm_occduke_wofail_n_2p')
    parser.add_argument('--DiffConfigPath', type=str, default='./config/diffusion.conf')  # configuration path for diffusion model
    parser.add_argument('--DataConfigPath', type=str, default='./config/data.yaml')  # configuration path for dataset
    parser.add_argument('--dataset_path', type=str, default='XXX/dataset/Occluded_Duke')  # dataset path
    parser.add_argument('--save_path', type=str, default='')  # output path # nohup revise
    parser.add_argument('--cond_scale', type=int, default=2)
    parser.add_argument('--guidance_prob', type=int, default=0.1)
    parser.add_argument('--sample_algorithm', type=str, default='ddim')  # ddpm, ddim
    parser.add_argument('--batch_size', type=int, default=64)  # 12G 16, 24G 64
    parser.add_argument('--save_wandb_logs_every_iters', type=int, default=50)
    parser.add_argument('--save_checkpoints_every_iters', type=int, default=10000)
    parser.add_argument('--save_wandb_images_every_epochs', type=int, default=10)  # 10
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    print ('Experiment: '+ args.exp_name)

    # load config
    DiffConf = DiffConfig(DiffusionConfig,  args.DiffConfigPath, args.opts, False)
    DataConf = DataConfig(args.DataConfigPath)

    DiffConf.training.ckpt_path = os.path.join(args.save_path, args.exp_name)
    DataConf.data.path = args.dataset_path

    if is_main_process():

        if not os.path.isdir(args.save_path): os.mkdir(args.save_path)
        if not os.path.isdir(DiffConf.training.ckpt_path): os.mkdir(DiffConf.training.ckpt_path)

    # DiffConf.ckpt = "checkpoints/last.pt"

    # for occ-duke
    num_classes= 702
    camera_num= 8
    view_num= 1


    use_id_loss=False
    id_loss_weight=1.0
    id_loss_k=1.0
    #  id_loss_weight * w(t)    w(t) = (exp(-k*t/T)-exp(-k)) / (1-exp(-k))

    # load reid model
    if use_id_loss:
        reid_model = make_model(cfg_reid_model, num_class=num_classes, camera_num=camera_num, view_num=view_num)
        
        cfg_reid_model.TEST.WEIGHT = "XXX/TransReID-main/log/occluded_duke/Occluded_Duke_idloss_add/diffusion_B/add_masks/time_100/mask_prob_100/transformer_120.pth"
        
        reid_model.load_state_dict(torch.load(cfg_reid_model.TEST.WEIGHT))
        reid_model.to(device)
        # frozen
        reid_model.eval()
        for p in reid_model.parameters():
            p.requires_grad = False
    
    begin_id_loss = 100

    main(settings=[args, DiffConf, DataConf], EXP_NAME=args.exp_name, local_rank=local_rank, is_distributed=is_distributed, use_id_loss=use_id_loss, id_loss_weight=id_loss_weight, id_loss_k=id_loss_k, begin_id_loss=begin_id_loss)