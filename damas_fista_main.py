import argparse
import sys
from torch.serialization import validate_cuda_device
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'      # 使用 GPU 3 
import time
from datasets.dataset import SoundDataset
from utils.utils import AverageMeter, save_model, pyContourf_two, Logger
from networks.damas_fista_net import DAMAS_FISTANet
from loss_function.losses import WingLoss

device_ids = [i for i in range(torch.cuda.device_count())]
def parse_option():
    parser = argparse.ArgumentParser(description='DAMAS_FISTANet for sound source in pytorch')

    parser.add_argument('--print_freq',
                            type=int,
                            default=1,
                            help='print frequency')
    parser.add_argument('--save_freq',
                            type=int,
                            default=10,
                            help='save frequency')
    parser.add_argument('--train_dir', 
                        help='The directory used to train the models',
                        default='./data/One_train.txt', type=str)
    parser.add_argument('--test_dir', 
                        help='The directory used to evaluate the models',
                        default='./data/One_val.txt', type=str)
    parser.add_argument('--label_dir', 
                        help='The directory used to evaluate the models',
                        default='./data/sound_source_data_fixed_distance/NewLabel/', type=str)
    parser.add_argument('--save_folder', dest='save_folder',
                        help='The directory used to save the models',
                        default='./models/',
                        type=str)
    parser.add_argument('--results_dir', 
                        help='The directory used to save the save image',
                        default='./img_results/', type=str)
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--val_epochs', default=9, type=int, metavar='N',
                        help='number of val epochs to run')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size for dataloader')
    parser.add_argument('--learning_rate', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate')
    parser.add_argument('--MultiStepLR',
                            action='store_true',
                            help='using MultiStepLR')
    parser.add_argument('--lr_decay_epochs',
                            type=str,
                            default='2, 4',
                            help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate',
                        type=float,
                        default=0.01,
                        help='decay rate for learning rate')                            
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--loss', type=str, default=None, choices=['Wing_loss', 'cross_entropy_loss', 'mse_loss'], help='loss')
    parser.add_argument('--LayNo', 
                        default=5, 
                        type=int,
                        help='iteration nums')

    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    record_time = time.localtime(time.time())
    args.model_name = 'DAMAS_FISTA-Net_lr_{}_decay_{}_bsz_{}_{}'.\
        format(args.learning_rate, args.weight_decay, args.batch_size, time.strftime('%m-%d-%H-%M', record_time))

    if args.MultiStepLR:
        args.model_name = '{}_MultiStepLR'.format(args.model_name)

    return record_time, args


# 加载声源数据
def set_loader(args):
    train_dataloader = torch.utils.data.DataLoader(
        SoundDataset(args.train_dir, args.label_dir),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(
        SoundDataset(args.test_dir, args.label_dir),
        batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True)

    return train_dataloader, test_dataloader



def set_model(args):
    # 加载模型
    model = DAMAS_FISTANet(args.LayNo)
    # 定义loss函数
    # criterion = torch.nn.L1Loss(reduction="mean") * 1e6
    if args.loss == 'Wing_loss':
        criterion = WingLoss()
        criterion.cuda()
    elif args.loss == 'cross_entropy_loss':
        criterion = nn.CrossEntropyLoss()
        criterion.cuda()

    elif args.loss == 'mse_loss':
        criterion = torch.nn.MSELoss()
        wandb.watch(model)

    else:
        criterion = None

    model.cuda()
    cudnn.benchmark = True

    return model, criterion


def set_optimizer(args, model):
    # 定义优化器
    # optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    return optimizer


def adjust_learning_rate(args, optimizer, epoch):
    # 定义学习率策略
    if args.MultiStepLR:
        # args.learning_rate *= 0.95 ** (epoch)
        args.learning_rate *= 0.95
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
            print('lr=', param_group['lr']) 
    



def train(train_dataloader, model, criterion, optimizer, epoch, args):
 
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for idx, (ATA, ATb, DAS_results, label, label_coordinate_to_vector_pos, _) in enumerate(train_dataloader):
        # x0 = torch.zeros(DAS_results.shape, dtype=torch.float64)
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            ATA = ATA.cuda(non_blocking=True)
            ATb = ATb.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            # x0 = x0.cuda(non_blocking=True)
            DAS_results = DAS_results.cuda(non_blocking=True)

        bsz = label.shape[0]

        # compute_loss
        # output = model(x0, ATA, ATb)
        output = model(DAS_results, ATA, ATb)
        print("==================output", torch.max(output))
        # print("==================label", torch.max(label))
        # print("==================torch.sqrt(output)", torch.sqrt(output))

        # SPL_output = 20 * torch.log10(2.2204e-16 + torch.sqrt(output) / 2e-5)
        # SPL_output = torch.clamp(SPL_output, min=0.0)
        # print("==================output", output)
        # SPL_label = 20 * torch.log10(2.2204e-16 + torch.sqrt(label) / 2e-5)
        # SPL_label = torch.clamp(SPL_label, min=0.0)
        # print("==================label", label)

        # SPL_label.shape_____[batch, 1681, 1]
        # center_points = torch.zeros(SPL_output.shape[0], 1)
        # for i in range(output.shape[0]):
        #     center_points[i] = (torch.squeeze(SPL_output[i], 1)[torch.squeeze(label_coordinate_to_vector_pos[i], 1).type(torch.long)] 
        #     - torch.squeeze(SPL_label[i], 1)[torch.squeeze(label_coordinate_to_vector_pos[i], 1).type(torch.long)]) ** 2

        loss = torch.sum((output - label) ** 2)

        # loss = torch.mean(center_points)
        # loss = 0.5 * torch.sum((SPL_output - SPL_label) ** 2) / torch.sum(SPL_label ** 2)
        # loss = 0.5 * torch.sum((output - label) ** 2) / torch.sum(label ** 2)
        # loss = criterion(SPL_output, SPL_label)
        # loss = torch.sum((SPL_output - SPL_label) ** 2)
        # loss = torch.mean((output - label) ** 2)
        loss = 1 * torch.sum((output - label) ** 2)
        # loss = torch.abs(torch.mean((output-label))) / 2e-5 / 2e-5
        losses.update(loss.item(), bsz)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # print("==================================grad")
        # print(model.lambda_step.grad)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        # print info
        if (idx + 1) % args.print_freq == 0:
           print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                      epoch,
                      idx + 1,
                      len(train_dataloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses))
        sys.stdout.flush()

    return losses.avg





def test(test_dataloader, model, criterion, args, record_time, epoch):
    """test"""
    # args.results_dir = args.results_dir + '{}__ckpt_epoch_{}/'.format(time.strftime('%m-%d-%H-%M', record_time), epoch) 
    # if not os.path.exists(args.results_dir):
    #     os.makedirs(args.results_dir)
    # filename = args.results_dir + 'loss.txt'
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (ATA, ATb, DAS_results, label, _, sample_name) in enumerate(test_dataloader):
            # x0 = torch.zeros(DAS_results.shape, dtype=torch.float64)
            ATA = ATA.cuda()
            ATb = ATb.cuda()
            DAS_results = DAS_results.cuda()
            label = label.cuda()
            # x0 = x0.cuda()
            bsz = label.shape[0]
            # forward
            output = model(DAS_results, ATA, ATb)
            SPL_output = 20 * torch.log10(2.2204e-16 + torch.sqrt(output) / 2e-5)
            SPL_output = torch.clamp(SPL_output, min=0.0)
            SPL_label = 20 * torch.log10(2.2204e-16 + torch.sqrt(label) / 2e-5)
            SPL_label = torch.clamp(SPL_label, min=0.0)

            # loss = criterion(SPL_output, SPL_label)
            # loss = torch.mean((SPL_output - SPL_label) ** 2)
            loss = torch.sum((output - label) ** 2)
            # np_loss = loss.cpu().numpy()
            # np_output = output.cpu().numpy()
            # np_label = label.cpu().numpy()

            print("##MAX_SPL_output", torch.max(SPL_output))
            print("####MAX_SPL_label", torch.max(SPL_label))

            # update metric
            losses.update(loss.item(), bsz)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
            if idx % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                                idx,
                                len(test_dataloader),
                                batch_time=batch_time,
                                loss=losses))

        return losses.avg






def main():
    record_time, args = parse_option()
    sys.stdout = Logger("train_info_debug_{}.txt".format(time.strftime('%m-%d-%H-%M', record_time)))
    args.save_folder = args.save_folder + '{}/'.format(time.strftime('%m-%d-%H-%M', record_time))
    if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

    # wandb.init(
    #     project='damas_fista_sound_source_location',
    #     entity='joaquin_chou',
    #     name="DAMAS_FISTANet" + args.model_name,
    #     config=args
    # )

    # build data loader
    train_dataloader, test_dataloader = set_loader(args)

    # build model and criterion
    model, criterion = set_model(args)

    # build optimizer
    optimizer = set_optimizer(args, model)

    print('===========================================')
    print('DAMAS_FISTA-Net...')
    print('===> Start Epoch {} End Epoch {}'.format(args.start_epoch, args.epochs + 1))
    # training routine
    for epoch in range(1, args.epochs + 1):
        if args.MultiStepLR:
            adjust_learning_rate(args, optimizer, epoch)
      
        # train for one epoch
        time1 = time.time()
        loss = train(train_dataloader, model, criterion, optimizer,
                                epoch, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        # wandb.log({"train_loss": loss, "epoch": epoch})

        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

    # evaluation
        if epoch % args.val_epochs  == 0:
            loss = test(test_dataloader, model, criterion, args, record_time, epoch)
            # wandb.log({"test_loss": loss, "epoch": epoch})

    # save the last model
    save_file = os.path.join(args.save_folder, 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)







if __name__ == '__main__':
    main()
