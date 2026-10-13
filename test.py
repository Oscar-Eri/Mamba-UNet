import os
from utils import *
from models.DAVT import DAVT_fomer as Net
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image, ImageDraw, ImageFont

# 设置文件夹路径
input_folder = "/hy-tmp/上传/lsui/yanLSUI/underwater"   # 输入图像所在文件夹
output_folder = "/hy-tmp/90yan/OO/results_LSUI"       # 结果保存文件夹
model_path = "/hy-tmp/DAVT/save_model/newstart/best_model_1.pth"  # 预训练模型路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 加载模型
use_gpu = torch.cuda.is_available()
model = Net().cuda() if use_gpu else Net()
checkpoint = torch.load(model_path, map_location="cuda" if use_gpu else "cpu")
model.load_state_dict(checkpoint['model'])
model.eval()

# 初始化指标列表
psnr_list = []
ssim_list = []

patchsize = 8  # 为确保输入尺寸为16的倍数

for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        imgPath = os.path.join(input_folder, filename)
        gt_path = os.path.join(input_folder.replace("underwater", "gt"), filename)

        # 加载图像（均转换为RGB）
        input_img = Image.open(imgPath).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        # 调整尺寸：保证宽高均为 patchsize 的倍数
        new_width = (input_img.width // patchsize // 2) * patchsize * 2
        new_height = (input_img.height // patchsize // 2) * patchsize * 2
        input_img = input_img.resize((new_width, new_height), Image.BILINEAR)
        gt_img = gt_img.resize((new_width, new_height), Image.BILINEAR)

        # 预处理输入图像，转换到[-1, 1]范围，并增加batch维度
        lr_img = convert_image(input_img, source='pil', target='[-1, 1]')
        lr_img.unsqueeze_(0)
        if use_gpu:
            lr_img = lr_img.cuda()

        # 生成增强图像
        with torch.no_grad():
            sr_tensor = model(lr_img).squeeze(0).cpu().detach()
            enhanced_img = convert_image(sr_tensor, source='[-1, 1]', target='pil')

        # 计算生成图像与GT图像的指标
        enhanced_np = np.array(enhanced_img)
        gt_np = np.array(gt_img)
        PSNR = peak_signal_noise_ratio(enhanced_np, gt_np)
        SSIM = ssim(enhanced_np, gt_np, win_size=3, channel_axis=-1)
        psnr_list.append(PSNR)
        ssim_list.append(SSIM)
        print(f"{filename} - PSNR: {PSNR:.2f}, SSIM: {SSIM:.4f}")

        # 拼接输入图像、生成图像和GT图像
        width, height = input_img.size
        composite_width = width * 3
        composite_height = height
        composite_img = Image.new("RGB", (composite_width, composite_height))
        # 左侧：输入图像；中间：生成图像；右侧：GT图像
        composite_img.paste(input_img, (0, 0))
        composite_img.paste(enhanced_img, (width, 0))
        composite_img.paste(gt_img, (width * 2, 0))

        # 添加标注
        draw = ImageDraw.Draw(composite_img)
        font = ImageFont.load_default()

        # 标注各区域标题
        draw.text((10, 10), "Input", fill=(255, 255, 255), font=font)
        draw.text((width + 10, 10), "Enhanced", fill=(255, 255, 255), font=font)
        draw.text((width * 2 + 10, 10), "GT", fill=(255, 255, 255), font=font)

        # 在Enhanced区域底部标注PSNR与SSIM指标
        metrics_text = f"PSNR: {PSNR:.2f}, SSIM: {SSIM:.4f}"
        # 使用draw.textbbox获取文本尺寸
        bbox = draw.textbbox((0, 0), metrics_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # 计算文本在Enhanced区域（中间区域）的居中位置
        x_text = width + (width - text_width) // 2
        y_text = height - text_height - 10
        draw.text((x_text, y_text), metrics_text, fill=(255, 255, 255), font=font)

        # 保存拼接结果
        save_path = os.path.join(output_folder, filename)
        composite_img.save(save_path)

# 输出平均指标
avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)
print("=" * 30)
print(f"平均 PSNR: {avg_psnr:.2f}")
print(f"平均 SSIM: {avg_ssim:.4f}")
