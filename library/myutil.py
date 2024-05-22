import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import kornia

class SelfAttention(nn.Module):
    """
    自注意力模块，用于计算输入特征图的注意力权重
    """
    def __init__(self, in_channels, hidden_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        batch_size, _, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width)
        key = self.key_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        energy = torch.einsum('bik,bjk->bij', query, key)
        attention = torch.softmax(energy, dim=-1)

        attention_out = torch.einsum('bij,bjk->bik', attention, value)
        attention_out = attention_out.view(batch_size, -1, height, width)

        out = self.gamma * attention_out + x
        return out

class DynamicWeightedLoss(nn.Module):
    """
    动态加权损失函数，使用自注意力机制动态调整损失项的权重
    """
    def __init__(self, in_channels, hidden_channels):
        super(DynamicWeightedLoss, self).__init__()
        self.attention = SelfAttention(in_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 1)
        # 提前创建 VGG19 模型和 Sobel 边缘检测器
        #self.vgg = models.vgg19(pretrained=True).features[:36].eval()
        #self.sobel = kornia.filters.Sobel()
    def ssim_loss(self,target, pre_loss, window_size=11, sigma=1.5):
        # 获取数据范围
        min_val = torch.min(target).item()
        max_val = torch.max(target).item()
        data_range = max_val - min_val

        # 图像的均值、方差和协方差
        channels = target.shape[1]
        weight = torch.ones(channels, channels, window_size, window_size).to(target.device) / (window_size ** 2)
        mu1 = F.conv2d(target, weight, padding=window_size // 2)
        mu2 = F.conv2d(pre_loss, weight, padding=window_size // 2)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2
        sigma1_sq = F.conv2d(target ** 2, weight, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(pre_loss ** 2, weight, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(target * pre_loss, weight, padding=window_size // 2) - mu12

        # SSIM计算
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        # SSIM损失
        ssim_loss = 1 - ssim_map

        return ssim_loss
    def forward(self, output, target, huber_c):
        huber_loss = 2 * huber_c * (torch.sqrt((output - target) ** 2 + huber_c**2) - huber_c)
        #output_features = self.vgg(output)
        #target_features = self.vgg(target)
        #perception_loss = F.mse_loss(output_features, target_features)

        ssim_losss = self.ssim_loss(target,output)

        #output_edges = self.sobel(output)
        #target_edges = self.sobel(target)
        #edge_loss = F.mse_loss(output_edges, target_edges)
        print("myutil——huber_loss:",huber_loss.shape)
        print("myutil——ssim_losss:",ssim_losss.shape)
        loss_values = torch.stack([huber_loss, ssim_losss], dim=1).unsqueeze(1)
        print("myutil——loss_values:",loss_values.shape)
        attention_out = self.attention(loss_values)
        print("myutil——1:",attention_out)
        attention_out = attention_out.mean(dim=[2, 3])
        print("myutil:",attention_out)
        weights = torch.softmax(self.fc(attention_out).view(-1), dim=0)
        print("myutil:weights",weights.shape)
        weighted_loss = (weights * loss_values)
        print("myutil:weighted_loss",weighted_loss.shape)
        return weighted_loss

# 提前创建 DynamicWeightedLoss 实例
def create_loss_weight(hidden_channels=64, in_channels=4):
    return DynamicWeightedLoss(in_channels=in_channels, hidden_channels=hidden_channels)

# 计算动态加权损失
def compute_dynamic_weights(weight_loss_fn, output, target, huber_c):
    return weight_loss_fn(output, target, huber_c)
