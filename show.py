import os
import sys
import cv2
import argparse
import math
import numpy as np
import itertools

import torch
from torch import nn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

import settings
from dataset import ShowDataset
from model import Restormer
from cal_ssim import SSIM
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = settings.device_id
logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)


# torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def PSNR(img1, img2):
    b, _, _, _ = img1.shape
    # mse=0
    # for i in range(b):
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)  # +mse
    if mse == 0:
        return 100
    # mse=mse/b
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class Session:
    def __init__(self):
        self.show_dir = settings.show_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.show_dir)
        ensure_dir(settings.model_dir)
        logger.info('set show dir as %s' % settings.show_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        if len(settings.device_id) > 1:
            self.net = nn.DataParallel(Restormer()).cuda()
        else:
            self.net = Restormer().cuda()
        self.ssim = SSIM().cuda()
        self.dataloaders = {}
        self.ssim = SSIM().cuda()
        self.a = 0
        self.t = 0

    def get_dataloader(self, dataset_name):
        dataset = ShowDataset(dataset_name)
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

    def inf_batch(self, name, batch, i):
        file_name = batch['file_name']
        file_name = str(file_name[0])
        O, B = batch['O'].cuda(), batch['B'].cuda()
        # O_gamma1, O_gamma2 = batch['O_gamma1'].cuda(), batch['O_gamma2'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        # O_gamma1, O_gamma2 = Variable(O_gamma1, requires_grad=False), Variable(O_gamma2, requires_grad=False)

        with torch.no_grad():
            import time
            t0 = time.time()
            b, c, h, w = O.size()
            # O = F.upsample(O, (1024,1024))
            img = self.net(O)
            # img = F.upsample(img, (h,w))
            Syn_low_syn_enhanced = img
            t1 = time.time()
            comput_time = t1 - t0
            print(comput_time)
            ssim = self.ssim(Syn_low_syn_enhanced, B).data.cpu().numpy()
            psnr = PSNR(Syn_low_syn_enhanced.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
            print('psnr:%4f-------------ssim:%4f' % (psnr, ssim))
            return Syn_low_syn_enhanced, psnr, ssim, file_name

    def save_image(self, No, imgs, name, file_name, model_epoch):
        for i, img in enumerate(imgs):
            img = (img.cpu().data * 255).numpy()
            img = np.clip(img, 0, 255)
            img = np.transpose(img, (1, 2, 0))
            h, w, c = img.shape

            dir = self.show_dir + '/' + str(model_epoch)
            ensure_dir(dir)

            img_file = os.path.join(dir, '%s.png' % (file_name))
            print(img_file)
            cv2.imwrite(img_file, img)


def run_show(ckp_syn_name):
    sess = Session()
    sess.load_checkpoints_net(ckp_syn_name)
    sess.net.eval()
    dataset = 'UHD_real'
    # dataset = 'Dense_Haze_NTIRE19_pair'
    # DICM
    # LIME
    # MEF
    # NPE
    # VV

    dt = sess.get_dataloader(dataset)

    # model_epoch = ckp_name.split('_')[1]
    model_epoch = dataset
    print('model_epoch', model_epoch)

    for i, batch in enumerate(dt):
        logger.info(i)
        if i > -1:
            imgs, psnr, ssim, file_name = sess.inf_batch('test', batch, i)
            sess.save_image(i, imgs, dataset, file_name, model_epoch)
            # sess.save_image(i, condifence[0], dataset, file_name, model_epoch + 'condidence-0')
            # sess.save_image(i, condifence[1], dataset, file_name, model_epoch + 'condidence-1')
            # sess.save_image(i, condifence[2], dataset, file_name, model_epoch + 'condidence-2')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m1', '--model1', default='latest_net')

    args = parser.parse_args(sys.argv[1:])

    run_show(args.model1)

