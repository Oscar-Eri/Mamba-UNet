
import torch
import cv2
import numpy as np
from utils import *
from torch import nn
class SSIM(torch.nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


class ComputteAvgSsim(torch.nn.Module):
    def __init__(self):
        super(ComputteAvgSsim, self).__init__()
        self.ssim=SSIM().cuda()
    def forward(self, img1, img2):
        ssim_reult=0
        B,_,_,_=img1.shape
        for i in range(B):
            raw=img1[i,::].cpu()
            label=img2[i,::].cpu()

            raw = np.array(convert_image(raw, source='[-1, 1]', target='pil'))
            label = np.array(convert_image(label, source='[-1, 1]', target='pil'))

            ssim_reult+=self.ssim(raw,label)

        return ssim_reult/B

class GradientLoss(torch.nn.Module):   #Laplacion 边缘感知损失
    def __init__(self):
        super(GradientLoss, self).__init__()
        Laplacian = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        Laplacian_3C = torch.Tensor(1, 3, 3, 3)
        Laplacian_3C[:, 0:3, :, :] = Laplacian
        self.conv_la = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_la.weight = torch.nn.Parameter(Laplacian_3C)
    def forward(self, X, Y):
        X_la = self.conv_la(X)
        Y_la = self.conv_la(Y)
        # compute gradient of Y
        self.conv_la.train(False)
        loss = nn.MSELoss()(X_la, Y_la)

        return loss

if __name__ == '__main__':
    ss=ComputteAvgSsim().cuda()

    imgPath = "E:/Image_enhancement/all_datasets/underwater_image/typeI_data/typeI/705_deep_3.5597_hori_9.97_.bmp"
    # imgPath =   "E:/数据集/水下图像/challenging-60/34.png"
    underwater_img =  Image.open(imgPath).convert('RGB')
    underwater_img=convert_image(underwater_img, source='pil', target='[-1, 1]')
    underwater_img.unsqueeze_(0)

    gt_name = imgPath.replace("underwater", "gt")
    gt_img = Image.open(gt_name).convert('RGB')
    gt_img = convert_image(gt_img, source='pil', target='[-1, 1]')
    gt_img.unsqueeze_(0)

    imgPath1 = "E:/Image_enhancement/all_datasets/underwater_image/typeII_data/type_II\\119_deep_3.3134_hori_11.8776_.bmp"
    # imgPath =   "E:/数据集/水下图像/challenging-60/34.png"
    underwater_img1 = Image.open(imgPath1).convert('RGB')
    underwater_img1 = convert_image(underwater_img1, source='pil', target='[-1, 1]')
    underwater_img1.unsqueeze_(0)

    gt_name1 = imgPath1.replace("underwater", "gt")
    gt_img1 = Image.open(gt_name1).convert('RGB')
    gt_img1 = convert_image(gt_img1, source='pil', target='[-1, 1]')
    gt_img1.unsqueeze_(0)

    underwater_imgcat=torch.cat([underwater_img,underwater_img1], 0).cuda()
    gt_imgcat = torch.cat([gt_img, gt_img1], 0).cuda()


    sss=ss(gt_imgcat,underwater_imgcat)

    print(sss)

