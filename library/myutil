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
        self.vgg = models.vgg19(pretrained=True).features[:36].eval()
        self.sobel = kornia.filters.Sobel()

    def forward(self, output, target, huber_c):
        huber_loss = 2 * huber_c * (torch.sqrt((output - target) ** 2 + huber_c**2) - huber_c).mean()
        
        output_features = self.vgg(output)
        target_features = self.vgg(target)
        perception_loss = F.mse_loss(output_features, target_features)

        ssim_loss = 1 - kornia.losses.ssim(output, target).mean()

        output_edges = self.sobel(output)
        target_edges = self.sobel(target)
        edge_loss = F.mse_loss(output_edges, target_edges)

        loss_values = torch.stack([huber_loss, perception_loss, ssim_loss, edge_loss])

        attention_out = self.attention(loss_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        attention_out = attention_out.mean(dim=[2, 3])
        print("myutil:",attention_out)
        weights = torch.softmax(self.fc(attention_out).view(-1), dim=0)
        
        weighted_loss = (weights * loss_values)

        return weighted_loss

# 提前创建 DynamicWeightedLoss 实例
def create_loss_weight(hidden_channels=64, in_channels=3):
    return DynamicWeightedLoss(in_channels=in_channels, hidden_channels=hidden_channels)

# 计算动态加权损失
def compute_dynamic_weights(weight_loss_fn, output, target, huber_c):
    return weight_loss_fn(output, target, huber_c)
