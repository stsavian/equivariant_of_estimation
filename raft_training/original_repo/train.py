from __future__ import print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"

import sys
sys.path.append('core')

import argparse

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from original_repo.core.raft import RAFT
import  original_repo.evaluate as evaluate
import original_repo.core.datasets as datasets
#import test_epe_and_imbalance.test_raft_function as imbalance_EPE_test
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def spatium_error(out_flo,target,spatium_norm=2):

    sim_m = torch.nn.functional.cosine_similarity(out_flo, target, dim=1,
                                              eps=1e-8)
    epsilon = 1e-8
    sim_clamped_m = torch.clamp(sim_m, -1 + epsilon, 1 - epsilon)
    theta_m = torch.acos(sim_clamped_m)
    gnd_mag = torch.norm(target, spatium_norm,1)
    s_m = torch.mul(gnd_mag,theta_m)
    s = torch.mean(s_m)
    #sim_m_clamped_deg = np.rad2deg(sim_clamped_m)
    #sim_deg = np.rad2deg(sim_clamped_m)
    #sim_deg = torch.mean(abs(sim_m_clamped_deg))

    return abs(s),abs(s_m)

def sequence_loss_with_spatium(flow_preds, flow_gt, valid, gamma=0.8,psi = 0,spatium_norm =2, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    epe_loss = 0.0
    spatium_flow_loss = 0.0
    #total_flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)


    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        epe_loss += i_weight * (valid[:, None] * i_loss).mean()
        s_loss = spatium_error(flow_preds[i],flow_gt,spatium_norm=spatium_norm)[1]
        spatium_flow_loss += i_weight * (valid[:, None] * s_loss).mean()

        #total_flow_loss += i_weight * (valid[:, None] * (i_loss + (s_loss * psi))).mean()
    total_flow_loss = epe_loss + psi* spatium_flow_loss


    spatium = spatium_error(flow_preds[-1],flow_gt,spatium_norm=spatium_norm)[0]
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    spatium_metrics = {
        's': spatium.mean().item(),
        '1px': (spatium < 1).float().mean().item(),
        '3px': (spatium < 3).float().mean().item(),
        '5px': (spatium < 5).float().mean().item(),
    }


    return total_flow_loss,epe_loss, metrics,spatium_flow_loss, spatium_metrics

def sequence_loss_with_imbalance(flow_preds,flow_preds_180,  flow_gt, valid, gamma=0.8, beta=0,norm='L1',max_flow=MAX_FLOW,average_epe_prediction=False):
    """ Loss function defined over sequence of flow predictions """

    #flow_preds_star = torch.rot90((flow_preds_180), 2, (2, 3))
    n_predictions = len(flow_preds)
    epe_loss = 0.0
    imbalance_flow_loss = 0.0
    total_flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        flow_preds_star = torch.rot90((flow_preds_180[i]), 2, (2, 3))
        i_weight = gamma**(n_predictions - i - 1)
        if average_epe_prediction:
            i_loss = ((flow_preds[i]-flow_preds_star)/2 - flow_gt).abs()
        else:
            i_loss = (flow_preds[i] - flow_gt).abs()
        if norm=='L1':
            i_imbalance_loss = (flow_preds[i] + flow_preds_star).abs()
        elif norm == 'L2':
            i_imbalance_loss = ((flow_preds[i] + flow_preds_star)**2).sqrt()
        elif norm=='stat_m2':
            i_imbalance_loss = (flow_preds[i] + flow_preds_star).abs()

        epe_loss += i_weight * (valid[:, None] * (i_loss)).mean()
        ## add here stat mean
        if norm=='stat_m2':
            imbalance_flow_loss += i_weight *(torch.sqrt((((i_imbalance_loss[:,0,...]* valid[:, None])**2).mean())+ \
                                                               torch.sqrt(((i_imbalance_loss[:, 1, ...]* valid[:, None] )** 2).mean()) ))
            total_flow_loss += (i_weight * (valid[:, None] * i_loss).mean() + (imbalance_flow_loss * beta))


        else:
            imbalance_flow_loss += i_weight * (valid[:, None] * i_imbalance_loss).mean()

            total_flow_loss += i_weight * (valid[:, None] * (i_loss+(i_imbalance_loss*beta))).mean()
        # if i == 11:
        #     print("EPE loss",i_loss.mean().item())
        #     print("IMBALANCE_loss",i_imbalance_loss.mean().item())
        #     print("TOTAL LOSS",total_flow_loss.mean().item())
        #total_flow_loss += i_weight * (valid[:, None] * (i_loss)).mean()

    # epe_loss =0
    # imbalance_flow_loss = 0
    #
    # euclidian_imbalance=0
    euclidian_imbalance = torch.sum((flow_preds[-1] + flow_preds_star)**2, dim=1).sqrt()
    euclidian_imbalance.view(-1)[valid.view(-1)]
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]


    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    imbalance_metrics = {
        'I': euclidian_imbalance.mean().item(),
        '1px': (euclidian_imbalance < 1).float().mean().item(),
        '3px': (euclidian_imbalance < 3).float().mean().item(),
        '5px': (euclidian_imbalance < 5).float().mean().item(),
    }
    #imbalance_metrics = {}
    return total_flow_loss,epe_loss, metrics,imbalance_flow_loss, imbalance_metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

# class Logger:
#     def __init__(self, model, scheduler,save_logs_path = None):
#         self.model = model
#         self.scheduler = scheduler
#         self.total_steps = 0
#         self.running_loss = {}
#         self.writer = None
#         self.save_logs_path = save_logs_path
#
#     def _print_training_status(self):
#         metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
#         training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
#         metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
#
#         # print the training status
#         print(training_str + metrics_str)
#
#         if self.writer is None:
#             self.writer = SummaryWriter(log_dir=self.save_logs_path)
#
#         for k in self.running_loss:
#             self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
#             self.running_loss[k] = 0.0
#
#     def push(self, metrics):
#         self.total_steps += 1
#
#         for key in metrics:
#             if key not in self.running_loss:
#                 self.running_loss[key] = 0.0
#
#             self.running_loss[key] += metrics[key]
#
#         if self.total_steps % SUM_FREQ == SUM_FREQ-1:
#             self._print_training_status()
#             self.running_loss = {}
#
#     def write_dict(self, results):
#         if self.writer is None:
#             self.writer = SummaryWriter()
#
#         for key in results:
#             self.writer.add_scalar(key, results[key], self.total_steps)
#
#     def close(self):
#         self.writer.close()


def train(args):
    save_folder_path = os.path.join(args.absolute_path, args.name)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    logs_path = os.path.join(save_folder_path,'tensorboard_logs')
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    #os.path.join(args.absolute_path, args.name)
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs' or args.stage != 'chairs2':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    #logger = Logger(model, scheduler,save_logs_path=logs_path)
    logger_writer = SummaryWriter(log_dir=logs_path)
    VAL_FREQ = args.val_freq
    add_noise = True


    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)
            ##### new stefano imbalance loss
            in1_r = image1.clone()
            in2_r = image2.clone()

            in1_r = torch.rot90(in1_r, 2, (2, 3))  # check for memory
            in2_r = torch.rot90(in2_r, 2, (2, 3))

            if args.double_fwd:
                if args.no_grad_on_rot_input:
                    with torch.no_grad():
                        output_r = model(in1_r, in2_r, iters=args.iters)
                else:
                    output_r = model(in1_r, in2_r, iters=args.iters)



                total_flow_loss,epe_loss, metrics,imbalance_flow_loss, imbalance_metrics = sequence_loss_with_imbalance(flow_predictions,output_r,\
                                                                                               flow, valid, args.gamma,beta= args.beta,norm=args.imb_train_norm,average_epe_prediction=args.train_on_average_epe)

                logger_writer.add_scalar('EPE_loss', epe_loss, total_steps)

                logger_writer.add_scalars('LOSS', {'total_flow_loss': total_flow_loss, \
                                                   'EPE_loss': epe_loss, 'imbalance_flow_loss': imbalance_flow_loss},total_steps)

                logger_writer.add_scalars('train', {'epe': metrics['epe'], 'I_L2': \
                    imbalance_metrics['I']}, total_steps)

                if args.average_loss:
                    loss = total_flow_loss/2
                else:
                    loss = total_flow_loss
                #logger.push(imbalance_metrics)
                # scaler.scale(total_flow_loss).backward()
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                #
                # scheduler.step()
                # scaler.step(optimizer)
                # scaler.update()





            else:
                #put previous here
                flow_predictions = model(image1, image2, iters=args.iters)
                if args.spatium_error == True:
                    total_flow_loss,epe_loss, metrics,spatium_flow_loss, spatium_metrics =\
                        sequence_loss_with_spatium(flow_predictions, flow, valid, args.gamma,psi = args.psi_spatium,spatium_norm=args.spatium_norm,max_flow=MAX_FLOW)
                    logger_writer.add_scalar('EPE_loss', epe_loss, total_steps)

                    logger_writer.add_scalars('LOSS', {'total_flow_loss': total_flow_loss, \
                                                     'EPE_loss': epe_loss,'spatium_flow_loss': spatium_flow_loss},total_steps)


                    logger_writer.add_scalars('train',{'epe': metrics['epe'],'spatium':\
                        spatium_metrics['s']},total_steps)
                    # logger.write_dict({'spatium_flow_loss': spatium_flow_loss})
                    # logger.write_dict({'spatium': spatium_metrics['s']})# stefano

                    loss=total_flow_loss

                else:
                    loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
                    logger_writer.add_scalar('EPE_loss',loss,total_steps)
                    logger_writer.add_scalar('EPE', metrics['epe'], total_steps)
                    #logger.write_dict({'epe_loss': loss})

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            #https://towardsdatascience.com/gradient-accumulation-overcoming-memory-constraints-in-deep-learning-36d411252d01
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            if total_steps % 200 == 0:
                print('step:', total_steps, ' loss: ',loss.item())
            #logger.push(metrics)
            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                #PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                PATH = os.path.join(args.absolute_path,args.name,'%d_%s.pth' % (total_steps + 1, args.name))
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        #results.update(evaluate.validate_chairs_with_imbalance(model.module))
                        results.update(evaluate.validate_chairs_with_imbalance(model.module,root=args.chairs_path))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel_with_imbalance(model.module,root=args.sintel_path))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module,root=args.kitti_path))

                #for key in results.keys():
                logger_writer.add_scalars('VAL',results,total_steps)


                #logger.push(metrics)

                #logger.write_dict(results['EPE chairs'])
                #logger.write_dict({'||I|| chairs':results['||I|| chairs']})
                #logger.write_dict(results)
                model.train()
                if args.stage != 'chairs' or args.stage != 'chairs2':
                    model.module.freeze_bn()
            


            if total_steps > args.num_steps:
                should_keep_training = False
                break

            # https: // stackoverflow.com / questions / 58216000 / get - total - amount - of - free - gpu - memory - and -available - using - pytorch
            #logger.write_dict({'memory reserved': r})
            # if total_steps % 4000 == 0 or (total_steps % 1000 ==0 & total_steps < 2000):
            #     r = torch.cuda.memory_reserved(0) / 1.e9
            #     #a = torch.cuda.memory_allocated(0) / 1.e9
            #     print(f"memory reserved {r:.3f} GB")
            #     print(f"memory allocated {a:.3f} GB")

            total_steps += 1

    logger_writer.close()
    #logger.close()
    #PATH = 'checkpoints/%s.pth' % args.name
    PATH = os.path.join(args.absolute_path,args.name, '%s.pth' % args.name)

    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--absolute_path', default='raft', help="save absolute path")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--chairs_path', default='/home/ssavian/training/FlyingChairs_release/data', help="flyingchairs path")
    parser.add_argument('--chairs2_path', default='/home/ssavian/training/FlyingChairs2',
                        help="flyingchairs path")
    parser.add_argument('--chairs2_flow_direction', type=str, default='all')
    parser.add_argument('--things_path', default='datasets/FlyingThings3D', help="flyingthings path")
    parser.add_argument('--things_flow_direction', type=str, default='all')

    parser.add_argument('--sintel_path', default='/home/ssavian/training/sintel/', help="sintel path")
    parser.add_argument('--kitti_path', default='/home/ssavian/', help="sintel path")
    parser.add_argument('--hd1k_path', default='/home/ssavian/', help="sintel path")

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--add_mirroring', action='store_true')
    parser.add_argument('--beta', type=float, default=0.0, help='imbalance loss  weighting')
    parser.add_argument('--double_fwd', action='store_true')
    parser.add_argument('--no_grad_on_rot_input', action='store_true')
    parser.add_argument('--average_loss', action='store_true', help="average loss when doing FWDG (grad accumulation)")


    parser.add_argument('--imb_train_norm',type=str, default='L1')
    parser.add_argument('--spatium_error', action='store_true')
    parser.add_argument('--psi_spatium', type=float, default=0.0, help='spatium loss  weighting')
    parser.add_argument('--spatium_norm', type=int, default=2)

    parser.add_argument('--plots_pth', default='/home/ssavian/training/FlyingChairs_release/data', help="flyingchairs path")
    parser.add_argument('--val_freq', type=int, default=5000)
    #parser.add_argument('--things_path', default='datasets/FlyingThings3D', help="flyingthings path")
    parser.add_argument('--train_on_average_epe', action='store_true')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train_path = train(args)

    # note = 'RAFT_' + train_path.split('/')[-1][:-4]
    # mode = 'clean'
    # #plots_pth = '/home/ssavian/training/plots/raft_trainings/'
    # #dataset_pth = '/home/ssavian/training/sintel/'
    # imbalance_EPE_test.test_raft_on_sintel(train_path, args.sintel_path, args.plots_pth, mode, note, rotate_sintel=False)
    # mode = 'final'
    # imbalance_EPE_test.test_raft_on_sintel(train_path, args.sintel_path, args.plots_pth, mode, note, rotate_sintel=False)