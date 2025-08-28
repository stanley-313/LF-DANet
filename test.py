import time
import argparse
import scipy.misc
import torch.backends.cudnn as cudnn
from utils.utils import *
from LF_DANet import Net
from tqdm import tqdm
import h5py
import numpy as np
from torchvision.transforms import ToTensor
import torch
from matplotlib import pyplot as plt
import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--model_name', type=str, default='wo_epi')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upscale_factor", type=int, default=4, help="upscale factor")
    parser.add_argument('--testset_dir', type=str, default='/wwj/datasets/LFSR/data_for_test/')
    parser.add_argument('--testdata', type=str, default='SR_5x5_4x/')
    parser.add_argument("--patchsize", type=int, default=256, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--stride", type=int, default=64,
                        help="The stride between two test patches is set to patchsize/2")
    parser.add_argument('--channels', type=int, default=60, help='channels')
    parser.add_argument('--model_path', type=str, default='./log/wo_epi/wo_epi_4xSR_5x5_epoch_61.pth.tar')
    # parser.add_argument('--model_path', type=str, default='../NTIRE23_LFSR_DistgEPIT/checkpoints/Exp80.DistgEPITv6.B4.L5e-5.E100.P128_32mp.S5.EMA0.999.FT.pth')
    parser.add_argument('--save_path', type=str, default='./Results/')

    return parser.parse_args()


def test(cfg, test_Names, test_loaders):
    net = Net(cfg.angRes, cfg.upscale_factor, cfg.channels)
    net.to(cfg.device)
    cudnn.benchmark = True

    if os.path.isfile(cfg.model_path):
        model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
        try:
            net.load_state_dict(model['state_dict'])
        except:
            net.load_state_dict({k.replace('module.netG.', ''): v for k, v in model['state_dict'].items()})
    else:
        print("=> no model found at '{}'".format(cfg.model_path))

    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_loaders[index]
            outLF, psnr_epoch_test, ssim_epoch_test = inference(test_loader, test_name, net)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (
                test_name, psnr_epoch_test, ssim_epoch_test))
            pass
        print(time.ctime()[4:-5] + ' Average_Result: Average_PSNR---%.6f, Average_SSIM---%.6f' % (
            (sum(psnr_testset) / len(psnr_testset)), (sum(ssim_testset) / len(ssim_testset))))
        pass


def inference(test_loader, test_name, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in tqdm((enumerate(test_loader)), total=len(test_loader), ncols=70):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angRes, w*angRes
        label = label.squeeze()

        uh, vw = data.shape
        h0, w0 = uh // cfg.angRes, vw // cfg.angRes
        subLFin = LFdivide(data, cfg.angRes, cfg.patchsize, cfg.stride)  # numU, numV, h*angRes, w*angRes
        numU, numV, H, W = subLFin.shape
        subLFout = torch.zeros(numU, numV, cfg.angRes * cfg.patchsize * cfg.upscale_factor,
                               cfg.angRes * cfg.patchsize * cfg.upscale_factor)

        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(
                    0)  # patchsize 128 tmp (1,1,640,640)  patchsize 32 tmp (1,1,160,160)
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    out = net(tmp.to(cfg.device))
                    subLFout[u, v, :, :] = out.squeeze()

        outLF = LFintegrate(subLFout, cfg.angRes, cfg.patchsize * cfg.upscale_factor, cfg.stride * cfg.upscale_factor,
                            h0 * cfg.upscale_factor, w0 * cfg.upscale_factor)

        psnr, ssim = cal_metrics(label, outLF, cfg.angRes)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        save_path = cfg.save_path + '/' + cfg.model_name + '/' + cfg.testdata

        isExists = os.path.exists(save_path + test_name)
        if not (isExists):
            os.makedirs(save_path + test_name)

        from scipy import io
        scipy.io.savemat(save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
                         {'LF': outLF.numpy()})
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return outLF, psnr_epoch_test, ssim_epoch_test

def test_one_img(img):
    net = Net(cfg.angRes, cfg.upscale_factor, cfg.channels)
    net.to(cfg.device)
    model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
    net.load_state_dict(model['state_dict'])
    cudnn.benchmark = True
    with h5py.File(img, 'r') as hf:
        data = np.array(hf.get('Lr_SAI_y'))
        label = np.array(hf.get('Hr_SAI_y'))
        data, label = np.transpose(data, (1, 0)), np.transpose(label, (1, 0))
        data, label = ToTensor()(data.copy()), ToTensor()(label.copy())
    with torch.no_grad():
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angRes, w*angRes
        label = label.squeeze()
        uh, vw = data.shape
        h0, w0 = uh // cfg.angRes, vw // cfg.angRes
        subLFin = LFdivide(data, cfg.angRes, cfg.patchsize, cfg.stride)  # numU, numV, h*angRes, w*angRes
        numU, numV, H, W = subLFin.shape
        subLFout = torch.zeros(numU, numV, cfg.angRes * cfg.patchsize * cfg.upscale_factor,
                               cfg.angRes * cfg.patchsize * cfg.upscale_factor)
        subLFout_featuremap_s = torch.zeros(numU, numV, cfg.angRes * cfg.patchsize,
                               cfg.angRes * cfg.patchsize)
        subLFout_featuremap_a = torch.zeros(numU, numV, cfg.angRes * cfg.patchsize,
                               cfg.angRes * cfg.patchsize)
        subLFout_featuremap_e = torch.zeros(numU, numV, cfg.angRes * cfg.patchsize,
                               cfg.angRes * cfg.patchsize)
        subLFout_featuremap = torch.zeros(numU, numV, cfg.angRes * cfg.patchsize,
                               cfg.angRes * cfg.patchsize)

        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)  # patchsize 128 tmp (1,1,640,640)  patchsize 32 tmp (1,1,160,160)
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    out = net(tmp.to(cfg.device))
                    fms = list(list(net.children())[2].children())[0][0].feature_map_s.squeeze().cpu()[0, :, :, :].view(5, 5, h0//4, w0//4).permute(0,2,1,3).contiguous().view(uh//4, vw//4)
                    fma = list(list(net.children())[2].children())[0][0].feature_map_a.squeeze().cpu()[0, :, :, :].view(5, 5, h0//4, w0//4).permute(0,2,1,3).contiguous().view(uh//4, vw//4)
                    fme = list(net.children())[3].feature_map_e.squeeze().cpu()[0, :, :, :].view(5, 5, h0//4, w0//4).permute(0,2,1,3).contiguous().view(uh//4, vw//4)
                    # fm = list(net.children())[2][0].feature_map.squeeze().cpu()[0, :, :, :].view(5, 5, h0//4, w0//4).permute(0,2,1,3).contiguous().view(uh//4, vw//4)
                    subLFout[u, v, :, :] = out.squeeze()
                    subLFout_featuremap_s[u, v, :, :] = fms
                    subLFout_featuremap_a[u, v, :, :] = fma
                    subLFout_featuremap_e[u, v, :, :] = fme
                    # subLFout_featuremap[u, v, :, :] = fm

        outLF = LFintegrate(subLFout, cfg.angRes, cfg.patchsize * cfg.upscale_factor, cfg.stride * cfg.upscale_factor,
                            h0 * cfg.upscale_factor, w0 * cfg.upscale_factor)
        outLF_feature_s = LFintegrate(subLFout_featuremap_s, cfg.angRes, cfg.patchsize, cfg.stride,
                            h0, w0)
        outLF_feature_a = LFintegrate(subLFout_featuremap_a, cfg.angRes, cfg.patchsize, cfg.stride,
                            h0, w0)
        outLF_feature_e = LFintegrate(subLFout_featuremap_e, cfg.angRes, cfg.patchsize, cfg.stride,
                             h0, w0)
        # outLF_feature = LFintegrate(subLFout_featuremap, cfg.angRes, cfg.patchsize, cfg.stride,
        #                     h0, w0)
        psnr, ssim = cal_metrics(label, outLF, cfg.angRes)
        print(psnr, ssim)
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(np.array(outLF_feature_s[2,2]), norm='linear')
        plt.subplot(2,2,2)
        plt.imshow(np.array(outLF_feature_a[2,2]), norm='linear')
        plt.subplot(2,2,3)
        plt.imshow(np.array(outLF_feature_e[2,2]), norm='linear')
        # plt.subplot(2,2,4)
        # plt.imshow(np.array(outLF_feature[2,2]))
        plt.colorbar()
        plt.show()

def main(cfg):
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    test(cfg, test_Names, test_Loaders)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
    # test_one_img('/home/wsz/sda1/xw/workspace/CSWinLFSR/data_for_test/SR_5x5_4x/Stanford_Gantry/Lego Knights.h5')
