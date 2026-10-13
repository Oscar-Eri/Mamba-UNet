import time
#https://blog.csdn.net/didiaopao/article/details/120276922  #troch==1.8.1
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import dataCollector
from datasets import SRDataset
import torch.nn as nn
from logger import Logger
import sys
from loss import GradientLoss
from models.DAVT import DAVT_fomer,init_weights
from utils import *
import pytorch_msssim
def train(train_loader, model, optimizer, use_gpu, mse, ssim,gradientLoss,batch_size,measure_loss=False):
    print("Training ...")
    model.train()
    running_loss= 0.
    training_loss = 0.

    mse_loss_3 = 0
    ssim_loss_3 = 0
    grad_loss_3 = 0.

    # 梯度权重
    weigh_mse = 5
    weigh_ssim = 30
    weigh_grad = 0.01#0.01–0.06

    #与原图的度量损失越大越好


    interval = 8
    # print("Batch_size:", batch_size * interval)
    start = time.time()

    for batch_idx, (raw_image, gt_image) in enumerate(train_loader):
        if use_gpu:
            inputs = raw_image.cuda()
            labels = gt_image.cuda()
        outputs = model(inputs)
        # 第三阶段
        loss_mse_3 = weigh_mse * mse(outputs, labels)
        loss_ssim_3 = weigh_ssim * (1 - ssim(outputs, labels))
        loss_grad_3 = weigh_grad * gradientLoss(outputs, labels)
        mse_loss_3 += loss_mse_3
        ssim_loss_3 += loss_ssim_3
        grad_loss_3 += loss_grad_3

        # 汇总所有损失
        loss_sum = loss_mse_3 + loss_ssim_3 + loss_grad_3

        loss_back = loss_sum / interval
        optimizer.zero_grad()  # zero the parameter gradients
        loss_back.backward()
        optimizer.step()  # 在这里面可以找到梯度,参数更新

        running_loss += loss_sum.item()
        training_loss+=loss_sum.item()

        if (batch_idx + 1) % interval == 0:
            # optimizer.step()  # 在这里面可以找到梯度,参数更新
            # optimizer.zero_grad() # zero the parameter gradients
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Loss_mse_3:{:.5f}  Loss_ssim_3:{:.5f} Loss_grad_3:{:.5f} Lr:{:.6f}".format(
                100. * (batch_idx + 1) / len(train_loader),
                end - start,
                training_loss / interval, #总损失
                mse_loss_3 / interval,  # 与原图的mse损失
                ssim_loss_3 / interval,  # 与原图的SSIM损失
                grad_loss_3 / interval,  # 与目标的梯度损失
                optimizer.state_dict()['param_groups'][0]['lr']
            ))
            training_loss = 0.
            mse_loss_3 = 0.
            ssim_loss_3 = 0.
            grad_loss_3=0.
            start = time.time()

    train_loss = running_loss /len(train_loader)
    print('train_loss:' + str(round(train_loss,4)))
    return train_loss

def val(val_loader, model, use_gpu,mse,ssim,gradientLoss,measure_loss=False):
    print("Testing ...")
    # model.half()
    model.eval()
    test_loss = 0.

    mse_loss_3 = 0
    ssim_loss_3 = 0
    grad_loss_3 = 0.

    # 梯度权重
    weigh_mse=5
    weigh_ssim = 30
    weigh_grad = 0.01  # 0.01–0.06


    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_loader):
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            # 第三阶段
            loss_mse_3 = weigh_mse * mse(outputs, labels)
            loss_ssim_3 = weigh_ssim * (1 - ssim(outputs, labels))
            loss_grad_3 = weigh_grad*gradientLoss(outputs, labels)
            mse_loss_3 += loss_mse_3
            ssim_loss_3 += loss_ssim_3
            grad_loss_3 += loss_grad_3

            loss_sum = loss_mse_3 + loss_ssim_3 + loss_grad_3

            test_loss += loss_sum.item()

        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f}  Loss_mse_3:{:.5f}  Loss_ssim_3:{:.5f} Loss_grad_3:{:.5f}".format(
            100. * (idx + 1) / len(val_loader),
            end - start,
            test_loss / len(val_loader),
            mse_loss_3 / len(val_loader),  # 与原图的mse损失
            ssim_loss_3 / len(val_loader),  # 与原图的SSIM损失
            grad_loss_3 / len(val_loader),  # 与目标的梯度损失

        ))
    return test_loss / len(val_loader)


# 定义画图函数
def matplot_loss(train_loss, val_loss,best,model_name):

    name=model_name.split("/")[1].split(".")[0]
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(name+str(round(best, 3)))
    #保存损失图片

    plt.savefig(model_name.replace("pth", "jpg"))
    plt.show()


def main():
    # 超参数
    use_gpu = torch.cuda.is_available()
    pin_memory = True if use_gpu else False
    model_dir = "/hy-tmp/DAVT/save_model/newstart"


    os.makedirs(model_dir, exist_ok=True)
    # 加载数据

    #file_dir ="E:/dataSets/underWater/typeI_data/underwater_typeI"
    file_dir = "/hy-tmp/上传/NEWlsui/LSUI/typeI_data/underwater_typeI/underwater"

    output_json = "datasets_"+file_dir.split("/")[-2]
    data_collector = dataCollector.DataCollector(file_dir,output_json)
    # 控制台输出
    sys.stdout = Logger(model_dir + '/loss1.log', sys.stdout)
    sys.stderr = Logger(model_dir + '/loss1.log_file', sys.stderr)

    best_loss = 2000
    epochs =270

    loss_train = []
    loss_val = []
    index=0
    #针对EUVP
    patch_sizes = [128,192,255,225]  # {128:22min,160:30min,192:45min,256:90min,320:240min,352:280min}
    beishu=1
    batch_sizes = [8*beishu,  4*beishu, 2*beishu,1*beishu]  # [内存--》128：3800，160：5300，192：5400，256：6000：320：5800，352：5800]
    # batch_sizes = [8,  4, 2, 1]  # [内存--》128：3800，160：5300，192：5400，256：6000：320：5800，352：5800]
    # epochss = [0, 1, 2, 3]
    epochss = [0, 100, 200,300]
    # epochss = [0, 50, 100,150]

    mse = nn.MSELoss()  # 均方损失
    ssim = pytorch_msssim.SSIM()# SSIM 损失
    gradientLoss = GradientLoss()
    model = DAVT_fomer()#model_attention7.Net_nums(3,3)
    model.apply(init_weights)
    if use_gpu:
         model.cuda()
         # 损失函数
         mse.cuda()
         ssim.cuda()
         gradientLoss.cuda()



    lr = 5e-5
    betas = (0.9, 0.999)
    weight_decay = 1e-4
    warmup_epoch = 10
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, (epochs - warmup_epoch), eta_min=1e-6)

    # warmup
    warmup_scheduler = newWarmUpLR(optimizer, warmup_epoch)


    model_name = model_dir + '/best_model_1.pth'
    model_final = model_dir + '/final_model.pth'

    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    start_epoch = 0
    if os.path.exists(model_name):
        checkpoint = torch.load(model_name)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['val_loss']
        print('加载 best_epoch {} 成功！'.format(checkpoint['epoch']))

    if os.path.exists(model_final):
        # http://t.zoukankan.com/lijianming180-p-12370491.html
        """
        如果模型的保存是为了恢复训练状态，那么可以考虑同时保存优化器optimizer的参数
        """
        checkpoint = torch.load(model_final)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_schedule'])  # 加载lr_scheduler
        start_epoch = checkpoint['epoch']
        index = checkpoint['index']

        #加载
        patch_size = patch_sizes[index-1]
        batch_size = batch_sizes[index-1]
        train_dataset = SRDataset(data_collector.train_datasets, 'train', patch_size, img_type='[-1, 1]')
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=pin_memory,
                                                   drop_last=True)

        val_dataset = SRDataset(data_collector.test_datasets, 'test', patch_size, img_type='[-1, 1]') #使用全图进行测试
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 pin_memory=pin_memory,
                                                 drop_last=False)

        print('加载 final_epoch {} 成功！'.format(start_epoch))

    measure_loss=False


    for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
        if epoch in epochss:
            index = index + 1
            patch_size = patch_sizes[index-1]
            batch_size = batch_sizes[index-1]

            train_dataset = SRDataset(data_collector.train_datasets, 'train', patch_size, img_type='[-1, 1]')
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       pin_memory=pin_memory,
                                                       drop_last=True)

            val_dataset = SRDataset(data_collector.test_datasets, 'test', patch_size, img_type='[-1, 1]')
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     pin_memory=pin_memory,
                                                     drop_last=False)


        lr= optimizer.state_dict()['param_groups'][0]['lr']
        print("\nEpoch : %d" % (epoch + 1))
        print("best loss：", round(best_loss, 4))
        print("Patch_size:", patch_size)
        print("Batch_size:", batch_size)
        print("LR", round(lr, 6))

        train_loss = train(train_loader, model,optimizer,use_gpu,mse,ssim,gradientLoss,batch_size,measure_loss=measure_loss)
        val_loss = train_loss#val(val_loader, model, use_gpu,mse,ssim,gradientLoss,measure_loss=measure_loss)


        # 将损失加入列表
        loss_train.append(train_loss)
        loss_val.append(val_loss)

        # 保存每一次的模型（方便挑选最好的）
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "val_loss": val_loss,
                 'lr_schedule': scheduler.state_dict(),"index":index}
        model_name_epoch = model_dir + '/epoch_' + str(epoch+1) + '_' + str(round(train_loss, 4)) + "_" + str(
            round(val_loss, 4)) + "_" + str(round(lr, 6)) + '.pth'
        # torch.save(state, model_name_epoch)  # 保存包括框架模型

        
        # 保存最后一次模型
        torch.save(state, model_final)


        if 240 <= epoch <= 270:
            torch.save(model.state_dict(), f"weights_epoch_{epoch}.pth")

            
        # 保存最好的模型权重
        if val_loss < best_loss:
            folder = 'save_model'
            if not os.path.exists(folder):
                os.mkdir('save_model')
            best_loss = val_loss
            print(f"save best model, 第{epoch+1}轮")
            # torch.save(model.state_dict(), model_name)
            # 保存模型
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1,
                     "val_loss": val_loss, 'lr_schedule': scheduler.state_dict()}
            torch.save(state, model_name)

        if epoch >=warmup_epoch:
            scheduler.step()
        else:
            warmup_scheduler.step()
    matplot_loss(loss_train, loss_val, best_loss,model_name)  # loss 画图

if __name__ == '__main__':
    main()