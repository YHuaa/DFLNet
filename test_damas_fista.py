from traceback import print_tb
from networks.damas_fista_net import DAMAS_FISTANet
from utils.utils import pyContourf_two, pyContourf
from datasets.dataset import SoundDataset
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'      # 使用 GPU 3 


def parse_option():
    parser = argparse.ArgumentParser('argument for testing')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--test_dir', 
                        help='The directory used to evaluate the models',
                        default='./data/One_test.txt', type=str)
    parser.add_argument('--results_dir', 
                        help='The directory used to save the save image',
                        default='./img_results/Two_val/', type=str)
    parser.add_argument('--label_dir', 
                        help='The directory used to evaluate the models',
                        default='./data/sound_source_data_fixed_distance/NewLabel/', type=str)
    parser.add_argument('--ckpt', type=str, 
                        default='./models/06-17-12-04/last.pth',
                        help='path to pre-trained model')
                        
    parser.add_argument('--LayNo', 
                        default=5, 
                        type=int,
                        help='iteration nums')
    parser.add_argument('--init',
                        action='store_true',
                        help='using no training model to test')

    args = parser.parse_args()

    args.results_dir = args.results_dir + args.ckpt.split('/')[-2] + '_' + args.ckpt.split('/')[-1].split('.')[0] + '/'
    if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)

    return args

# 加载声源数据
def set_loader(args):
    test_dataloader = torch.utils.data.DataLoader(
        SoundDataset(args.test_dir, args.label_dir),
        batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True)

    return test_dataloader


def set_model(args):
    # 加载模型
    model = DAMAS_FISTANet(args.LayNo)
    criterion = torch.nn.MSELoss()

    if not args.init:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        state_dict = ckpt['model']
        if torch.cuda.is_available():
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
            model = model.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True
            model.load_state_dict(state_dict)
    else:
            model = model.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True

    return model, criterion



def test(test_dataloader, model, criterion, args):
    filename = args.results_dir + 'loss.txt'
    # time_count = []
    model.eval()

    with torch.no_grad():
        for idx, (ATA, ATb, DAS_results, label, _, sample_name) in enumerate(test_dataloader):
            # x0 = torch.zeros(DAS_results.shape, dtype=torch.float64)
            ATA = ATA.cuda()
            ATb = ATb.cuda()
            label = label.cuda()
            DAS_results = DAS_results.cuda()
            # x0 = x0.cuda()
            output = model(DAS_results, ATA, ATb)
            # output = model(x0, ATA, ATb)

            SPL_output = 20 * torch.log10(2.2204e-16 + torch.sqrt(output) / 2e-5)
            SPL_output = torch.clamp(SPL_output, min=0.0)
            print(max(max(SPL_output)))
            SPL_label = 20 * torch.log10(2.2204e-16 + torch.sqrt(label) / 2e-5)
            SPL_label = torch.clamp(SPL_label, min=0.0)


            
            loss = criterion(SPL_output, SPL_label)
            # np_loss = loss.cpu().numpy()
            # np_output = SPL_output.cpu().numpy()
            # np_label = SPL_label.cpu().numpy()

            # loss = criterion(output, label)
            np_loss = loss.cpu().numpy()
            np_output = output.cpu().numpy()
            np_label = label.cpu().numpy()

            print("####np.max(np_output)=", np.max(np_output))
    
            print("####np.max(np_label)=", np.max(np_label))

            with open(filename, 'a') as file_object:
                file_object.write(
                    '{}_np_L2_loss__{:4f}\n'.format(idx, np_loss))

            np_label = np_label.reshape(41, 41, order='F')
            np_output = np_output.reshape(41, 41, order='F')
            np.savetxt(r'test.txt',np_output, fmt='%8.2f', delimiter=',')
            pyContourf_two(np_output, np_label, args.results_dir, sample_name[0])
            # pyContourf(np_label)

def main():
    args = parse_option()
    model, criterion = set_model(args)
    # build data loader
    test_dataloader = set_loader(args)
    test(test_dataloader, model, criterion, args)

if __name__ == '__main__':
    main()