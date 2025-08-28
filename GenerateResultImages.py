import numpy as np
import h5py
import os
from imageio import imsave
from scipy.io import loadmat
from utils.imresize import *

sourceDataPath = r'D:\BaiduNetdiskDownload\LFSRdata\raw_data'
sourceDatasets = os.listdir(sourceDataPath)
resultsPath = r'D:\BaiduNetdiskDownload\LFSRdata\SR_5x5_x4\SR_5x5_x4\wo_epi\SR_5x5_4x'
SavePath = r'D:\BaiduNetdiskDownload\LFSRdata\SR_5x5_x4\SR_5x5_x4\LGT-LFSR'
model_name = 'GT'

# sourceDataPath = '/wwj/datasets/LFSR/data_for_test/SR_5x5_4x'
# sourceDatasets = os.listdir(sourceDataPath)
# resultsPath = './Results/wo_epi/SR_5x5_4x'
# SavePath = sourceDataPath + '/SRimgaes'

angRes = 5

def rgb2ycbcr(x):
    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  65.481 * x[:, :, 0] + 128.553 * x[:, :, 1] +  24.966 * x[:, :, 2] +  16.0
    y[:,:,1] = -37.797 * x[:, :, 0] -  74.203 * x[:, :, 1] + 112.000 * x[:, :, 2] + 128.0
    y[:,:,2] = 112.000 * x[:, :, 0] -  93.786 * x[:, :, 1] -  18.214 * x[:, :, 2] + 128.0

    y = y / 255.0
    return y


def ycbcr2rgb(x):
    mat = np.array(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]])
    mat_inv = np.linalg.inv(mat)
    offset = np.matmul(mat_inv, np.array([16, 128, 128]))
    mat_inv = mat_inv * 255

    y = np.zeros(x.shape, dtype='double')
    y[:,:,0] =  mat_inv[0,0] * x[:, :, 0] + mat_inv[0,1] * x[:, :, 1] + mat_inv[0,2] * x[:, :, 2] - offset[0]
    y[:,:,1] =  mat_inv[1,0] * x[:, :, 0] + mat_inv[1,1] * x[:, :, 1] + mat_inv[1,2] * x[:, :, 2] - offset[1]
    y[:,:,2] =  mat_inv[2,0] * x[:, :, 0] + mat_inv[2,1] * x[:, :, 1] + mat_inv[2,2] * x[:, :, 2] - offset[2]
    return y

# local
for datasetName in sourceDatasets:

    gtFolder = os.path.join(sourceDataPath, datasetName, 'test')
    sceneFiles = os.listdir(gtFolder)
    resultsFolder = os.path.join(resultsPath, datasetName)

    for scene in sceneFiles:
        sceneName = scene.split('.')[0]
        print(f'Generating result images of Scene_{sceneName} in Dataset {datasetName} ......')
        gtPath = os.path.join(gtFolder, scene)
        resultPath = os.path.join(resultsFolder, sceneName + '.mat')

        try:
            data = h5py.File(gtPath, 'r')
            LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
        except:
            data = loadmat(gtPath)
            LF = np.array(data[('LF')])

        try:
            data = h5py.File(resultPath, 'r')
            outLF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
        except:
            data = loadmat(resultPath)
            outLF = np.array(data[('LF')])

        (U, V, H, W) = outLF.shape
        print(LF.shape, outLF.shape)
        # Extract central angRes * angRes views
        LF = LF[(U - angRes) // 2:(U + angRes) // 2, (V - angRes) // 2:(V + angRes) // 2, :H, :W, 0:3]
        LF = LF.astype('single')



        for u in range(U):
            for v in range(V):
                Hr_SAI = LF[u, v, :, :, :]
                sr_sai_y = outLF[u, v, :, :]
                Hr_SAI_CbCr = rgb2ycbcr(Hr_SAI)[:, :, 1:3]
                # hr_sai_cbcr = imresize(lr_SAI_CbCr, 4)
                hr_sai_ycbcr = np.concatenate([sr_sai_y.reshape(H, W, -1), Hr_SAI_CbCr], axis=-1)
                hr_sai_grb = ycbcr2rgb(hr_sai_ycbcr)
                savePath = os.path.join(SavePath, datasetName, sceneName)
                os.makedirs(savePath, exist_ok=True)
                data = np.array(hr_sai_grb.clip(0, 1) * 255, dtype='uint8')
                imsave(os.path.join(savePath, 'View_' + str(u) + '_' + str(v) + '.bmp'), data)

# remote
# for datasetName in sourceDatasets:
#
#     gtFolder = os.path.join(sourceDataPath, datasetName)
#     sceneFiles = os.listdir(gtFolder)
#     resultsFolder = os.path.join(resultsPath, datasetName)
#
#     for scene in sceneFiles:
#         sceneName = scene.split('.')[0]
#         print(f'Generating result images of Scene_{sceneName} in Dataset {datasetName} ......')
#         gtPath = os.path.join(gtFolder, scene)
#         resultPath = os.path.join(resultsFolder, sceneName + '.mat')
#
#         try:
#             data = h5py.File(gtPath, 'r')
#             LF = np.array(data[('Sr_SAI_cbcr')])
#             (c, ha, wa) = LF.shape
#             LF = LF.reshape(c, angRes, ha//angRes, angRes, wa//angRes).transpose((3,1,4,2,0))
#         except:
#             data = loadmat(gtPath)
#             LF = np.array(data[('Sr_SAI_cbcr')])
#             (ha, wa, c) = LF.shape
#             LF = LF.reshape(angRes, ha//angRes, angRes, wa//angRes, c).transpose((0,2,1,3,4))
#
#         try:
#             data = h5py.File(resultPath, 'r')
#             outLF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
#         except:
#             data = loadmat(resultPath)
#             outLF = np.array(data[('LF')])
#
#         (U, V, H, W) = outLF.shape
#         print(LF.shape, outLF.shape)
#         # Extract central angRes * angRes views
#         LF = LF[(U - angRes) // 2:(U + angRes) // 2, (V - angRes) // 2:(V + angRes) // 2, :H, :W, :]
#         outLF = outLF.astype('double')
#         LF = LF.astype('double')
#
#
#
#         for u in range(U):
#             for v in range(V):
#                 Hr_SAI_CbCr = LF[u, v, :, :, :]
#                 sr_sai_y = outLF[u, v, :, :]
#                 # Hr_SAI_CbCr = rgb2ycbcr(Hr_SAI)[:, :, 1:3]
#                 # hr_sai_cbcr = imresize(lr_SAI_CbCr, 4)
#                 hr_sai_ycbcr = np.concatenate([sr_sai_y.reshape(H, W, -1), Hr_SAI_CbCr], axis=-1)
#                 hr_sai_grb = ycbcr2rgb(hr_sai_ycbcr)
#                 savePath = os.path.join(SavePath, datasetName, sceneName)
#                 os.makedirs(savePath, exist_ok=True)
#                 data = np.array(hr_sai_grb.clip(0, 1) * 255, dtype='uint8')
#                 imsave(os.path.join(savePath, 'View_' + str(u) + '_' + str(v) + '.bmp'), data)