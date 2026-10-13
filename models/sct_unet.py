from re import T
import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F
import time
#import torchsummary as summary

import torch.profiler

cfg_Ideatormer = {
    "position_bias":False,#是否使用位置编码
    "num_blocks" :[2, 4, 6, 8], #TransformerBlock numbers
    "channels":[8, 16, 32, 64],
    "num_heads" : [2, 4, 8, 16],
    "num_refinement":3, #last TransformerBlock numbers
    "patchsizes":[32, 16, 8, 4] # TransformerBlock patchsizes
}

def init_weights(m):
    """
    weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        # nn.init.kaiming_normal_(m.weight, mode="fan_out")
        nn.init.normal_(m.weight, 0.0,0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def window_partition(x, patch_szie: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    """
    # x_cpu=x.cpu().detach().numpy()
    B,num_head,C ,H, W= x.shape
    x = x.view(B, num_head, C,patch_szie, H//patch_szie,  patch_szie, W //patch_szie).permute(0, 1, 3,5,2, 4, 6).contiguous()

    # x1_cpu = x.cpu().detach().numpy()


    windows=x.view(B, num_head, patch_szie*patch_szie, -1)
    return windows

def window_reverse(x, patch_szie: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    """
    B, num_head= x.shape[0], x.shape[1]
    out = x.view(B, num_head,  patch_szie,  patch_szie, -1,H//patch_szie,W//patch_szie)
    windows=out.permute(0, 1, 4 ,2,5,3,6).contiguous().view(B, num_head,-1,H,W)
    return windows

def channelAttention(featureMap,deepMap):
    assert featureMap.shape==deepMap.shape
    B, C, H, W = featureMap.shape

    featureMapK = featureMap.reshape(B, C, H * W)
    deepMap_q = deepMap.reshape(B, C, H * W)

    q, k = F.normalize(featureMapK, dim=-1), F.normalize(deepMap_q, dim=-1)
    attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()), dim=-1)

    out_featureMap = torch.matmul(attn, featureMapK).reshape(B, -1, H, W) + featureMap
    out_deepMap = torch.matmul(attn, deepMap_q).reshape(B, -1, H, W) + deepMap

    return out_featureMap * out_deepMap+out_featureMap


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.channels=channels
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
    def forward(self, x):
        return self.body(x)


class DeepMapDownSample(nn.Module):
    def __init__(self):
        super(DeepMapDownSample, self).__init__()
        self.body = nn.Sequential(nn.PixelUnshuffle(2))
    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels,rate=2):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * rate, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(rate))
    def forward(self, x):
        return self.body(x)

class SeparableConv2D(nn.Module):
    """
    深度可分离卷积3*3
    "Depthwise conv + Pointwise conv"
    """
    def __init__(self, in_channels, out_channels):
        super(SeparableConv2D, self).__init__()
        # Depthwise 过程
        self.DWconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)#深度可分离卷积
        # Pointwise 过程
        self.PWconv= nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # x_input = x.cpu().detach().numpy()
        x = self.DWconv(x)
        x = self.PWconv(x)
        # x_2 = x.cpu().detach().numpy()
        return x


class GFFN(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(GFFN, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = torch.relu(out)
        out = self.excitation(out)
        w=torch.sigmoid(out)
        out = out * w
        return out


class SMDHA(nn.Module):
    """
    spatial multi-head attention 空间多头注意力,位置关系（消融），（最大，平均池化）增加可学习参数
    """
    def __init__(self,channels=48, num_heads=4,patchsize=32,position_bias=False):
        super(SMDHA, self).__init__()
        self.num_heads = num_heads
        self.channels=channels
        self.patchsize=patchsize

        self.position_bias=position_bias

        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        # self.pos_embed = nn.Parameter(torch.zeros(1, channels, h//window_size,w//window_size))
        self.qkv_PW = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)# PW卷积
        self.qkv_DW= nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False) #DW卷积

        self.project_out = SeparableConv2D(channels, channels) #深度可分离卷积
        self.q_avg = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.q_max = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.k_avg = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.k_max = nn.Parameter(torch.ones(1, num_heads, 1, 1))


        #     添加可学习的位置信息
        #     1.绝对位置？
        if position_bias:
            self.position_bias_table = nn.Parameter(
                torch.zeros(self.patchsize * self.patchsize, self.patchsize * self.patchsize, num_heads))  # 学习位置编码

            nn.init.trunc_normal_(self.position_bias_table, std=.02)

    def forward(self, x):
        # x1_cpu = self.position_bias_table.permute(2, 0,1).cpu().detach().numpy()
        B, C, H, W= x.shape
        q, k, v = self.qkv_DW(self.qkv_PW(x)).chunk(3, dim=1)

        q = q.reshape(B, self.num_heads, -1, H,W)
        k = k.reshape(B, self.num_heads, -1, H, W)
        v = v.reshape(B, self.num_heads, -1, H, W)

        q_avg = torch.mean(q, dim=2, keepdim=True).squeeze(dim=2)
        q_max, _ = torch.max(q, dim=2, keepdim=True)
        q_max=q_max.squeeze(dim=2)
        q = (self.q_avg * q_avg + self.q_max * q_max).unsqueeze(dim=2)


        k_avg = torch.mean(k, dim=2, keepdim=True).squeeze(dim=2)
        k_max, _ = torch.max(k, dim=2, keepdim=True)
        k_max = k_max.squeeze(dim=2)
        k = (self.k_avg * k_avg + self.k_max * k_max).unsqueeze(dim=2)

        q = window_partition(q, self.patchsize)
        k = window_partition(k, self.patchsize)
        v = window_partition(v, self.patchsize)
        # q_cpu = q.cpu().detach().numpy()
        # q_cpu = q.cpu().detach().numpy()

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)  # 归一化

        if self.position_bias:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature + self.position_bias_table.permute(2, 0,1).contiguous().unsqueeze(0)
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature

        attn = torch.softmax(attn, dim=-1)  # 1，4，64，64

        out = torch.matmul(attn, v)

        # 重构
        out = window_reverse(out, self.patchsize, H, W).view(B, -1,H,W)
        out = self.project_out(out)
        return out

class CMDHA(nn.Module):
    def __init__(self, channels=48, num_heads=4,patchsize=32):
        super(CMDHA, self).__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.patchsize = patchsize
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv_PW = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)  # PW卷积
        self.qkv_DW = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)  # DW卷积

        self.avepool = nn.AdaptiveAvgPool2d((patchsize, patchsize))  # 自适应平均池化
        self.maxpool = nn.AdaptiveMaxPool2d((patchsize, patchsize))  # 自适应最大池化
        self.q_avg = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.q_max = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.k_avg = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.k_max = nn.Parameter(torch.ones(1, num_heads, 1, 1))


        self.project_out = SeparableConv2D(channels, channels)  # 深度可分离卷积


    def forward(self, x):
        B, C, H, W = x.shape

        q, k, v = self.qkv_DW(self.qkv_PW(x)).chunk(3, dim=1)

        # 自适应池化
        q_avg = self.avepool(q)
        q_avg = q_avg.reshape(B, self.num_heads, -1, self.patchsize * self.patchsize)
        q_max = self.maxpool(q)
        q_max = q_max.reshape(B, self.num_heads, -1, self.patchsize * self.patchsize)

        k_avg = self.avepool(k)
        k_avg = k_avg.reshape(B, self.num_heads, -1, self.patchsize * self.patchsize)
        k_max = self.maxpool(k)
        k_max = k_max.reshape(B, self.num_heads, -1, self.patchsize * self.patchsize)
        q = self.q_avg * q_avg + self.q_max * q_max
        k = self.k_avg * k_avg + self.k_max * k_max

        v = v.reshape(B, self.num_heads, -1, H* W)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(B, -1, H, W))
        return out

class TransformerBlock(nn.Module):
    def __init__(self,channels, num_heads, patchsize=16):
        super(TransformerBlock, self).__init__()

        # 空间注意力模块
        self.norm1 = nn.LayerNorm(channels)
        self.attn1 = SMDHA(channels, num_heads,patchsize=patchsize,position_bias=cfg_Ideatormer["position_bias"])
        self.norm2 = nn.LayerNorm(channels)
        # self.ffn1 = GDFN(channels, expansion_factor) #resfomer论文中的块
        self.ffn1 = GFFN(channels, channels//4) #resfomer论文中的块

        # 通道注意力模块
        self.norm3 = nn.LayerNorm(channels)
        self.attn2 = CMDHA(channels, num_heads, patchsize=patchsize)
        self.norm4 = nn.LayerNorm(channels)
        # self.ffn2 = GDFN(channels, expansion_factor)
        self.ffn2 = GFFN(channels, channels//4)



    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.attn1(self.norm1(x.reshape(B, C, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(B, C, H, W))
        x = x + self.ffn1(self.norm2(x.reshape(B, C, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(B, C, H, W))

        x = x + self.attn2(self.norm3(x.reshape(B, C, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(B, C, H, W))
        x = x + self.ffn2( self.norm4(x.reshape(B, C, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(B, C,H, W))
        return x


class SCTUNet(nn.Module):
    # def __init__(self,num_blocks=[1, 2, 3, 4], num_heads=[2,4,8,16], channels=[8, 16, 32, 64], num_refinement=2,patchsizes=[32, 16, 8, 4]):
    def __init__(self,num_blocks=cfg_Ideatormer["num_blocks"], num_heads=cfg_Ideatormer["num_heads"], channels=cfg_Ideatormer["channels"],
                 num_refinement=cfg_Ideatormer["num_refinement"],patchsizes=cfg_Ideatormer["patchsizes"]):
        super(SCTUNet, self).__init__()
        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)


        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(num_ch, num_ah, patchsize=patchsize)
                                    for _ in range(num_tb)])
                                       for num_tb, num_ah, num_ch ,patchsize in
                                       zip(num_blocks, num_heads, channels,patchsizes)])

        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])



        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[i], num_heads[i], patchsize=patchsizes[i])
                                                       for _ in range(num_blocks[i])])
                                                            for i in [2,1,0]])

        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])

        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False) for i in
                                      reversed(range(1, len(channels)))])


        # deepmap下采样
        self.downs_map = DeepMapDownSample()  # nn.ModuleList([DeepMapDownSample() for num_ch in channels[:-1]])
        # 深度图采样
        self.map = nn.ModuleList([nn.Conv2d(4 ** i, channels[i], kernel_size=3, padding=1, bias=False) for i in range(3)])


        # 输出层
        self.output1 = SeparableConv2D(channels[0]*2, 3)
        self.tanh = nn.Tanh()

        # out_dec2 上采样，深度可分离卷积还原
        self.Up2=UpSample(channels[1])
        self.output2 = SeparableConv2D(channels[0]*2, 3)

        # out_dec3 上采样，深度可分离卷积还原
        self.Up3= UpSample(channels[2],rate=4)
        self.output3 = SeparableConv2D(channels[0]*2, 3)



    def forward(self, x,deep_map):
        fo = self.embed_conv(x) # C,H,W
        '''
        encoder
        '''
        # encoder stage1
        out_enc1 = self.encoders[0](fo) #C,H,W
        out_enc1=channelAttention(out_enc1, self.map[0](deep_map))#C,H,W

        # encoder stage2
        deep_map2=self.downs_map(deep_map)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))#2C,H//2,W//2
        out_enc2 = channelAttention(out_enc2, self.map[1](deep_map2))  # 2C,H//2,W//2

        # encoder stage3
        deep_map3 = self.downs_map(deep_map2)
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))#4C,H//4,W//4
        out_enc3 = channelAttention(out_enc3, self.map[2](deep_map3))  # 4C,H//4,W//4

        # encoder stage4
        deep_map4 = self.downs_map(deep_map3)
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))#8C,H//8,W//8
        out_enc4 = channelAttention(out_enc4, deep_map4)  # 8C,H//8,W//8

        '''
        decoder
        '''

        # decoder stage3
        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))#4C,H//4,W//4

        # decoder stage2
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))#2C,H//2,W//2
        # decoder stage1
        out_dec1 = self.decoders[2](self.reduces[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1)))#2C,H,W
        '''
        output
        '''
        # output stage1
        y1 =   self.tanh(self.output1(torch.cat([out_dec1, fo], dim=1)))+x #唯一改变就是将底层特征embed_conv给添加了回来
        # output stage2
        y2 =  self.tanh(self.output2(torch.cat([self.Up2(out_dec2), fo], dim=1)))+ x  # 唯一改变就是将底层特征embed_conv给添加了回来
        # # output stage3
        y3 =  self.tanh(self.output2(torch.cat([self.Up3(out_dec3),fo], dim=1)))+ x  # 唯一改变就是将底层特征embed_conv给添加了回来

        y1 = torch.clamp(y1, -1, 1, out=None)
        y2 = torch.clamp(y2, -1, 1, out=None)
        y3 = torch.clamp(y3, -1, 1, out=None)

        return y1 ,y2,y3





if __name__ == '__main__':
    torch.manual_seed(123)
    INPUT_SIZE=320
    input = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE-32).cuda()
    input1 = torch.randn(2, 1, INPUT_SIZE, INPUT_SIZE-32).cuda()
    print("模型输入", input.size())
    net = SCTUNet()
    net.cuda()
    net.eval()
    print("#打印模型")
    # print(net)
    out = net(input,input1)
    print("模型输出", out[0].size())
    #
    print("统计模型参数量")
    net1 = SCTUNet()
    flops, params = profile(net1, (torch.randn(1, 3, 128, 128),torch.randn(1, 1, 128, 128)))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f GFLOPs, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: ", total)
    print("Number of parameter: M", total / 1e6)

    # summary.summary(net, [(3, 256, 256),(1, 256, 256)])


    # 性能瓶颈分析
    # https://www.jianshu.com/p/1d09e56abbf8
    # Warn-up

    # input = torch.randn(1, 3, 160, 160).cuda()
    # input1 = torch.randn(1, 1, 160, 160).cuda()
    # for _ in range(5):
    #     start = time.time()
    #     outputs = net(input,input1)
    #     torch.cuda.synchronize()
    #     end = time.time()
    #     print('Time:{}ms'.format((end - start) * 1000))
    #
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True,
    #                                      profile_memory=True) as prof:
    #     outputs = net(input,input1)
    # print(prof.table())
    # prof.export_chrome_trace('sss/Ideatormer_profile.json')
