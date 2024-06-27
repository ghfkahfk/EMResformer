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
from model_map import ODE_DerainNet
from cal_ssim import SSIM

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
            self.net = nn.DataParallel(ODE_DerainNet()).cuda()
            # self.l2 = nn.DataParallel(MSELoss(),settings.device_id)
            # self.l1 = nn.DataParallel(nn.L1Loss(),settings.device_id)
            # self.ssim = nn.DataParallel(SSIM(),settings.device_id)
            # self.vgg = nn.DataParallel(VGG(),settings.device_id)
        else:
            self.net = ODE_DerainNet().cuda()
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

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def inf_batch(self, name, batch, i):
        # self.print_network(self.net)
        file_name = batch['file_name']
        print('file-name', file_name)
        file_name = str(file_name[0])
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        with torch.no_grad():
            import time
            t0 = time.time()
            out1,mask = self.net(O)
            t1 = time.time()
            comput_time = t1 - t0
            print(comput_time)
            ssim = self.ssim(out1, B).data.cpu().numpy()
            psnr = PSNR(out1.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
            print('psnr:%4f-------------ssim:%4f' % (psnr, ssim))
            return out1, psnr, ssim, file_name,mask

    def heatmap(self, img):
        if len(img.shape) == 3:
            print('=3')
            b, h, w = img.shape
            heat = np.zeros((b, 3, h, w)).astype('uint8')
            for i in range(b):
                heat[i, :, :, :] = np.transpose(cv2.applyColorMap(img[i, :, :], cv2.COLORMAP_JET), (2, 0, 1))
        else:
            print('!=3')
            b, c, h, w = img.shape
            print(c)
            heat_map = []
            for i in range(b):
                for j in range(c):
                    heat = np.zeros((b, 3, h, w)).astype('uint8')
                    heat[i, :, :, :] = np.transpose(cv2.applyColorMap(img[i, j, :, :], cv2.COLORMAP_JET), (2, 0, 1))
                    heat_map.append(heat)
        return heat_map

    def save_image(self, No, imgs, name, psnr, ssim, file_name):
        for i, img in enumerate(imgs):
            img = (img.cpu().data * 255).numpy()
            img = np.clip(img, 0, 255)
            img = np.transpose(img, (1, 2, 0))
            h, w, c = img.shape

            img_file = os.path.join(self.show_dir, '%s.png' % (file_name))
            print(img_file)
            cv2.imwrite(img_file, img)

    def save_mask(self, No, imgs, name, psnr, ssim, file_name):
        for i, img in enumerate(imgs):
            img = (img.cpu().data).numpy()
            img = np.clip(img, 0, 255)
            img = np.transpose(img, (1, 2, 0))
            h, w, c = img.shape

            img_file = os.path.join(self.show_dir, '%s.png' % (file_name))
            print(img_file)
            cv2.imwrite(img_file, img)

    def save_image_feature(self, dir, imgs):
        ensure_dir(dir)
        for i, img in enumerate(imgs):  # i 代表第几个卷积层
            img = img.cpu().data.numpy()
            print(img.shape)
            # img=img.squeeze(0)
            # print(img.shape)
            for j in range(settings.channel):  # j代表卷积层的第几个feature map
                print(img[j])
                a = (img[j] * 255)  # .numpy()
                a = np.clip(a, 0, 255)
                img_file = os.path.join(dir, '%d.png' % (j + 1))
                print(img_file)
                cv2.imwrite(img_file, a)

    # def run_show(ckp_name):
    #     sess = Session()
    #     sess.load_checkpoints(ckp_name)
    #     sess.net.eval()
    #     dataset = 'map'
    #     dt = sess.get_dataloader(dataset)
    #
    #     for i, batch in enumerate(dt):
    #         logger.info(i)
    #         imgs, filename, psnr, ssim, show = sess.inf_batch('test', batch)
    #         pool1 = show[0]
    #         pool2 = show[1]
    #         pool4 = show[2]
    #
    #         sess.save_image_feature('pool1', pool1)
    #         sess.save_image_feature('pool2', pool2)
    #         sess.save_image_feature('pool4', pool4)


def run_show(ckp_name):
    sess = Session()
    sess.load_checkpoints(ckp_name)
    sess.net.eval()
    dataset = 'featuremap'
    dt = sess.get_dataloader(dataset)

    for i, batch in enumerate(dt):
        logger.info(i)
        if i > -1:
            imgs, psnr, ssim, file_name, mask = sess.inf_batch('test', batch, i)
            mask1 = mask[0]
            sess.save_image_feature('../hazemask/1/', mask1)
            # mask_ = []
            # for i in range(settings.channel):
            #     # mask1 = np.transpose(mask_list1[i][0], (1, 2, 0))
            #     mask1 = mask_list1
            #     print('mask1.shape',mask1.shape)
            #     # mask_.append(mask1)
            #
            #     cv2.imwrite('../hazefreemask/1/%d.png'%i, mask1)

            mask2 = mask[1].cpu().data
            mask2 = mask2 * 255
            mask_list2 = np.clip(mask2.numpy(), 0, 255).astype('uint8')
            # mask_list2 = sess.heatmap(mask2)
            for i in range(settings.channel):
                mask2 = np.transpose(mask_list2[i][0], (1, 2, 0))
                ensure_dir('../hazefreemask/2/')
                cv2.imwrite('../hazefreemask/2/%d.png'%i, mask2)

            mask3 = mask[2].cpu().data
            mask3 = mask3 * 255
            mask_list3 = np.clip(mask3.numpy(), 0, 255).astype('uint8')
            # mask_list3 = sess.heatmap(mask3)
            for i in range(settings.channel):
                mask3 = np.transpose(mask_list3[i][0], (1, 2, 0))
                ensure_dir('../hazefreemask/3/')
                cv2.imwrite('../hazefreemask/3/%d.png' % i, mask3)

            mask4 = mask[3].cpu().data
            mask4 = mask4 * 255
            mask_list4 = np.clip(mask4.numpy(), 0, 255).astype('uint8')
            # mask_list4 = sess.heatmap(mask4)
            for i in range(settings.channel):
                mask4 = np.transpose(mask_list4[i][0], (1, 2, 0))
                ensure_dir('../hazefreemask/4/')
                cv2.imwrite('../hazefreemask/4/%d.png' % i, mask4)

            mask5 = mask[4].cpu().data
            mask5 = mask5 * 255
            mask_list5 = np.clip(mask5.numpy(), 0, 255).astype('uint8')
            # mask_list5 = sess.heatmap(mask5)
            for i in range(settings.channel):
                mask5 = np.transpose(mask_list5[i][0], (1, 2, 0))
                ensure_dir('../hazefreemask/5/')
                cv2.imwrite('../hazefreemask/5/%d.png' % i, mask5)

            mask6 = mask[5].cpu().data
            mask6 = mask6 * 255
            mask_list6 = np.clip(mask6.numpy(), 0, 255).astype('uint8')
            # mask_list6 = sess.heatmap(mask6)
            for i in range(settings.channel):
                mask6 = np.transpose(mask_list6[i][0], (1, 2, 0))
                ensure_dir('../hazefreemask/6/')
                cv2.imwrite('../hazefreemask/6/%d.png' % i, mask6)

            mask7 = mask[6].cpu().data
            mask7 = mask7 * 255
            mask_list7 = np.clip(mask7.numpy(), 0, 255).astype('uint8')
            # mask_list7 = sess.heatmap(mask7)
            for i in range(settings.channel):
                mask7 = np.transpose(mask_list7[i][0], (1, 2, 0))
                ensure_dir('../hazefreemask/7/')
                cv2.imwrite('../hazefreemask/7/%d.png' % i, mask7)

            mask8 = mask[7].cpu().data
            mask8 = mask8 * 255
            mask_list8 = np.clip(mask8.numpy(), 0, 255).astype('uint8')
            # mask_list8 = sess.heatmap(mask8)
            for i in range(settings.channel):
                mask8 = np.transpose(mask_list8[i][0], (1, 2, 0))
                ensure_dir('../hazefreemask/8/')
                cv2.imwrite('../hazefreemask/8/%d.png' % i, mask8)

            mask9 = mask[8].cpu().data
            mask9 = mask9 * 255
            mask_list9 = np.clip(mask9.numpy(), 0, 255).astype('uint8')
            # mask_list9 = sess.heatmap(mask9)
            for i in range(settings.channel):
                mask9 = np.transpose(mask_list9[i][0], (1, 2, 0))
                ensure_dir('../hazefreemask/9/')
                cv2.imwrite('../hazefreemask/9/%d.png' % i, mask9)

            # mask4 = mask[3].cpu().data
            # mask4 = mask4 * 255
            # mask4 = np.clip(mask4.numpy(), 0, 255).astype('uint8')
            # mask_list4 = sess.heatmap(mask4)
            # for i in range(settings.channel):
            #     mask4 = np.transpose(mask_list4[i][0], (1, 2, 0))
            #     ensure_dir('../mask/fusion21/')
            #     cv2.imwrite('../mask/fusion21/%d.png' % i, mask4)
            #
            # mask5 = mask[4].cpu().data
            # mask5 = mask5 * 255
            # mask5 = np.clip(mask5.numpy(), 0, 255).astype('uint8')
            # mask_list5 = sess.heatmap(mask5)
            # for i in range(settings.channel):
            #     mask5 = np.transpose(mask_list5[i][0], (1, 2, 0))
            #     ensure_dir('../mask/fusion12/')
            #     cv2.imwrite('../mask/fusion12/%d.png' % i, mask5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='latest_net')

    args = parser.parse_args(sys.argv[1:])

    run_show(args.model)

