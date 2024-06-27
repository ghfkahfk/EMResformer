import os
import sys
import cv2
import argparse
import numpy as np
import math
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch
_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')

# loss_fn_vgg = lpips.LPIPS(net='vgg')

import settings
from dataset import TestDataset
from model import Restormer
from cal_ssim import SSIM

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

os.environ['CUDA_VISIBLE_DEVICES'] = settings.device_id
logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
#torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
def PSNR(img1, img2):
    b,_,_,_=img1.shape
    #mse=0
    #for i in range(b):
    img1=np.clip(img1,0,255)
    img2=np.clip(img2,0,255)
    mse = np.mean((img1/ 255. - img2/ 255.) ** 2)#+mse
    if mse == 0:
        return 100
    #mse=mse/b
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)
        if len(settings.device_id) > 1:
            self.net = nn.DataParallel(Restormer()).cuda()
        else:
            self.net = Restormer().cuda()

        self.lpips = lpips
        self.l2 = MSELoss().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.ssim = SSIM().cuda()
        self.dataloaders = {}

    def ssim_gray(self, imgA, imgB, gray_scale=True):
        if gray_scale:
            score, diff = ssim(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY), full=True, multichannel=True)
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        else:
            score, diff = ssim(imgA, imgB, full=True, multichannel=True)
        return score

    def psnr_gray(self, imgA, imgB):
        psnr_val = psnr(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY))
        return psnr_val

    def get_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)
        self.dataloaders[dataset_name] = \
            DataLoader(dataset, batch_size=1,
                       shuffle=False, num_workers=1)
        return self.dataloaders[dataset_name]

    def load_checkpoints_net(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            logger.info('Load checkpoint %s' % ckp_path)
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])


    def inf_batch(self, name, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        # O_gamma1, O_gamma2 = batch['O_gamma1'].cuda(), batch['O_gamma2'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        # O_gamma1, O_gamma2 = Variable(O_gamma1, requires_grad=False), Variable(O_gamma2, requires_grad=False)

        with torch.no_grad():
            Syn_low_syn_enhanced = self.net(O)

        d = self.lpips(2*torch.clip(Syn_low_syn_enhanced.cpu(),0,1)-1, 2*B.cpu()-1)
        print('lpips',d)

        gt = B[0].data.cpu().numpy() * 255
        gt = np.clip(gt, 0, 255)
        gt = np.transpose(gt, (1, 2, 0))

        img = Syn_low_syn_enhanced[0].data.cpu().numpy() * 255
        img = np.clip(img, 0, 255)
        img = np.transpose(img, (1, 2, 0))

        l1_loss = self.l1(Syn_low_syn_enhanced, B)
        ssim = self.ssim_gray(img.astype(np.uint8), gt.astype(np.uint8))
        psnr = self.psnr_gray(img.astype(np.uint8), gt.astype(np.uint8))
        losses = {'L1 loss' : l1_loss }
        ssimes = {'ssim' : ssim}
        losses.update(ssimes)

        # img = np.clip(img, 0, 255)
        # img = np.transpose(img, (1, 2, 0))


        return losses, psnr,d


def run_test(ckp_syn_name):
    sess = Session()
    sess.load_checkpoints_net(ckp_syn_name)
    sess.net.eval()
    dt = sess.get_dataloader('test')
    psnr_all = 0
    d_all = 0
    all_num = 0
    all_losses = {}
    for i, batch in enumerate(dt):
        losses, psnr,d = sess.inf_batch('test', batch)
        psnr_all = psnr_all+psnr
        d_all = d_all + d

        batch_size = batch['O'].size(0)
        all_num += batch_size
        for key, val in losses.items():
            if i == 0:
                all_losses[key] = 0.
            all_losses[key] += val * batch_size
            logger.info('batch %d mse %s: %f' % (i, key, val))


    for key, val in all_losses.items():
        logger.info('totala mse %s: %f' % (key, val / all_num))


    print('psnr_ll_tea:%8f' % (psnr_all / all_num))
    print('lpips_ll_tea:%8f' % (d_all / all_num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model1', default='net_best_epoch')

    args = parser.parse_args(sys.argv[1:])

    run_test(args.model1)

