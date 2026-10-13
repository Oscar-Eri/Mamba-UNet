from utils import *

from models.DAVT import DAVT_fomer as Net
import torch
import numpy as np
import matplotlib.pyplot as plt  # plt 用于显示图片
from skimage.metrics import structural_similarity,peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim


if __name__ == '__main__':
    #加载模型
    # crop_size=256
    caijian=True
    patchsize=16
    #加载图片

    imgPath =   "E:/datasets/Ucolor/underwater/6_img_.png"
    #imgPath =  "E:/数据集/水下图像/合成数据集/typeIII_data/underwater_type_III/280_deep_3.207_hori_8.8023_.bmp"
    #
    # imgPath =   "E:/datasets/Ucolor/Ucolor_underwater/100_deep_3.0825_hori_7.5228_.png"
    #imgPath =   "E:/数据集/水下图像/raw-890-s/raw-890/raw-890_underwater/4_img_.png"


    underwater_img = Image.open(imgPath).convert('RGB')
    gt_name = imgPath.replace("underwater", "gt")
    gt_img = Image.open(gt_name).convert('RGB')


    if underwater_img.width%patchsize!=0:
        Width=(underwater_img.width//patchsize//2)*patchsize*2
    else:
        Width=underwater_img.width
    if underwater_img.height%patchsize!=0:
        Heigh=(underwater_img.height//patchsize//2)*patchsize*2
    else:
        Heigh = underwater_img.height
    underwater_img=underwater_img.resize((Width, Heigh),Image.BILINEAR)
    gt_img = gt_img.resize((Width, Heigh),Image.BILINEAR)


    #根据图片来判断模型输入图像尺寸
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = Net().cuda()#model_attention7.Net_nums(3,3)
    model_name = r'E:\dataSets\savemodel\best_model.pth'


    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model'])
    model_all = model_name.replace(".", "_all.")
    torch.save(model, model_all)
    # 保存模型
    model.eval()


    gt_img = np.array(gt_img)
    lr_img = convert_image(underwater_img, source='pil', target='[-1, 1]')

    lr_img.unsqueeze_(0)
    lr_img = lr_img.cuda()



    #测试
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        sr_img.save('test_srgan.jpg')
        result = np.array(sr_img)



    plt.figure(figsize=[20, 20])
    plt.subplot(1, 3, 1)
    plt.imshow(np.array(underwater_img))
    plt.title('raw')
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(result)
    plt.title('result')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(gt_img)
    plt.axis("off")
    plt.title('gt_img')
    plt.savefig("xxx.jpg")
    plt.show()

    #与结果图
    PSNR = peak_signal_noise_ratio(result, gt_img)
    SSIM = ssim(result, gt_img, win_size=3, channel_axis=-1)
    MSE = np.mean(np.square(result - gt_img))
    print("结果图MSE:", MSE)
    print("结果图PSNR:", PSNR)
    print("结果图SSIM:", SSIM)

    # 原图
    print("*"*23)
    oral=np.array(underwater_img)
    PSNR_oral = peak_signal_noise_ratio(result, oral)
    SSIM_oral = ssim(result, oral, win_size=3, channel_axis=-1)  # 指定 win_size=3
    MSE_oral = np.mean(np.square(result - oral))
    print("原图MSE:", MSE_oral)
    print("原图PSNR:", PSNR_oral)
    print("原图SSIM:", SSIM_oral)

    print(1)