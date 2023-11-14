import scipy.io as scio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'      # 使用 GPU 3 
import torch
import numpy as np
from utils.utils import data_preprocess
from networks.damas_fista_net import DAMAS_FISTANet
from utils.utils import pyContourf_two
import time

# test_real_data_path = 'D:/Ftp_Server/zgx/data/sound_source_data_fixed_distance/NewDataset/aa.mat'
# test_real_data_path = '/home/zhangyh/project/SoundNet/UNetAndFISTANet/data/sound_source_data_fixed_distance/RealTest/aa.mat'
test_real_data_path = './data/sound_source_data_fixed_distance/RealData/10K_Two/859.mat'
results_dir = './img_results/real_test/'
ckpt = './models/06-17-12-04/last.pth'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
sample_name = test_real_data_path.split('/')[-1].split('.')[0]

# LayNo = 5
LayNo = 5
add_Noise = True

def addnoise(original,snr):
        # print(len(original))
        N = np.random.randn(original.shape[0], original.shape[1]).astype(np.float32)
        numerator = sum(np.square(original.astype(np.float32)))
        denominator = sum(np.square(N))
        factor = 10**(snr/10.0)
        K = (numerator/(factor*denominator))**0.5
        noise = original + K*N
        return noise

if __name__ == '__main__':

    # f = h5py.File(test_real_data_path, 'r')
    # raw_sound_data = np.array(f['time_data'][()])
    # print('data_shape:', data.shape)
    # # data.shape = [9600, 64]
    raw_sound_data = scio.loadmat(test_real_data_path)['a'][:6000]
    # 是否加噪声测试
    if add_Noise:
        raw_sound_data = addnoise(raw_sound_data, snr=0)
    print(raw_sound_data.shape)
    _, ATA, ATb, DAS_results, label_coordinate_to_vector_pos = data_preprocess(raw_sound_data, None)
    DAS_result = DAS_results.reshape(41, 41, order='F')
    ATA = ATA.reshape(1, ATA.shape[0], ATA.shape[1])
    ATb = ATb.reshape(1, ATb.shape[0], ATb.shape[1])
    DAS_results = DAS_results.reshape(1, DAS_results.shape[0], DAS_results.shape[1])
    
    
     # 加载模型
    model = DAMAS_FISTANet(LayNo)
    ckpt = torch.load(ckpt, map_location='cpu')
    state_dict = ckpt['model']
    if torch.cuda.is_available():
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()
        # criterion = criterion.cuda()
        # cudnn.benchmark = True
        model.load_state_dict(state_dict)
    # criterion = torch.nn.MSELoss()

    start_time = time.time()
    # filename = results_dir + 'loss.txt'

    model.eval()
    with torch.no_grad():
        idx = 1
    # for idx, (ATA, ATb, DAS_results, label, _, sample_name) in enumerate(test_dataloader):
        # x0 = torch.zeros(DAS_results.shape, dtype=torch.float64)
        ATA = torch.tensor(ATA).cuda()
        ATb = torch.tensor(ATb).cuda()
        print(ATA.shape)
        print(ATb.shape)
        # label = label.cuda()
        DAS_results = torch.tensor(DAS_results).cuda()
        # x0 = x0.cuda()
        start = time.time()
        output = model(DAS_results, ATA, ATb)
        end = time.time()
        print(end - start)
        # output = model(x0, ATA, ATb)

        SPL_output = 20 * torch.log10(2.2204e-16 + torch.sqrt(output) / 2e-5)
        SPL_output = torch.clamp(SPL_output, min=0.0)
        # SPL_label = 20 * torch.log10(2.2204e-16 + torch.sqrt(label) / 2e-5)
        # SPL_label = torch.clamp(SPL_label, min=0.0)

        
        # loss = criterion(SPL_output, SPL_label)
        # np_loss = loss.cpu().numpy()
        # np_output = SPL_output.cpu().numpy()
        # np_label = SPL_label.cpu().numpy()

        # loss = criterion(output, label)
        # np_loss = loss.cpu().numpy()
        np_output = output.cpu().numpy()
        # np_label = label.cpu().numpy()

        print("####np.max(np_output)=", np.max(np_output))

        # print("####np.max(np_label)=", np.max(np_label))

        # with open(filename, 'a') as file_object:
        #     file_object.write(
        #         '{}_np_L2_loss__{:4f}\n'.format(idx, np_loss))

        # np_label = np_label.reshape(41, 41, order='F')
        end_time = time.time()
        print(end_time-start_time)
        np_output = np_output.reshape(41, 41, order='F')
        # scio.savemat('859_B1.mat',{'B':DAS_result})
        # scio.savemat('859_0_B1.mat',{'B':np_output})
        zero = np.zeros((41, 41))
        pyContourf_two(np_output, DAS_result, results_dir, sample_name)


    