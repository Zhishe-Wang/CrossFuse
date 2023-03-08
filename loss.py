from torch import nn
import torch

class g_content_loss(nn.Module):
    def __init__(self):
        super(g_content_loss, self).__init__()
        self.L2_loss = nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss(reduction="mean")
        self.gradient=gradient()
    def forward(self, img_ir, img_vi, img_fusion):
        lambda_2=1
        lambda_3=10
        image_vi_grad = self.gradient(img_vi)
        image_ir_grad = self.gradient(img_ir)
        image_fusion_grad = self.gradient(img_fusion)
        image_max_grad = torch.round((image_vi_grad + image_ir_grad) // (
                torch.abs(image_vi_grad + image_ir_grad) + 0.0000000001)) * torch.max(
            torch.abs(image_vi_grad), torch.abs(image_ir_grad))
        grad_loss = self.L1_loss(image_fusion_grad, image_max_grad)

        intensity_loss = self.L2_loss(img_fusion,img_ir)+lambda_2*self.L1_loss(img_fusion,img_vi)
        texture_loss =grad_loss
        content_loss = intensity_loss + lambda_3*texture_loss

        return content_loss,  intensity_loss , texture_loss

class gradient(nn.Module):
    def __init__(self,channels=1):
        super(gradient, self).__init__()
        laplacian_kernel = torch.tensor([[1/8,1/8,1/8],[1/8,-1,1/8],[1/8,1/8,1/8]]).float()

        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        self.laplacian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False
    def forward(self,x):
        return self.laplacian_filter(x) ** 2