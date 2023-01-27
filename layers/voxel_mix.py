import torch.nn as nn
import torch

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class voxel_mix_net(nn.Module):
    # [4,240,128,128] -> [4,80,128,128]
    def __init__(self):
        super(voxel_mix_net, self).__init__()
        in_channels = 80*6
        out_channels = 80
        local_global_channels = 80*3
        mix_channels = 80*2
        stride = 1
        self.se = SELayer(in_channels)
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
        
        self.gl_se = SELayer(local_global_channels)
        self.gl_residual = nn.Sequential(
            nn.Conv2d(local_global_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.gl_shortcut = nn.Sequential(
                nn.Conv2d(local_global_channels, out_channels, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu2 = nn.ReLU(inplace=True)

        # self.mix_se = SELayer(mix_channels)
        # self.mix_residual = nn.Sequential(
        #     nn.Conv2d(mix_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels)
        # )
        # self.mix_shortcut = nn.Sequential(
        #         nn.Conv2d(mix_channels, out_channels, stride=stride, kernel_size=1, bias=False),
        #         nn.BatchNorm2d(out_channels)
        #     )
        # self.relu3 = nn.ReLU(inplace=True)

    def forward(self, input):
        local_depth = input[:,:-80*3,:,:]
        lc_feature = self.se(local_depth)
        lc_feature = self.relu(self.residual_function(lc_feature) + self.shortcut(lc_feature))
        
        glob_depth = input[:,-80*3:,:,:]
        gl_feature = self.gl_se(glob_depth)
        gl_feature = self.relu2(self.gl_residual(gl_feature) + self.gl_shortcut(gl_feature))
        
        # mix_feature = torch.cat([lc_feature,gl_feature],dim=1)
        # result = self.mix_se(mix_feature)
        # result = self.relu2(self.mix_residual(result) + self.mix_shortcut(mix_feature))
        # return result
        gl_feature = self.gl_se(glob_depth)
        gl_feature = self.relu2(self.gl_residual(gl_feature) + self.gl_shortcut(gl_feature))
        
        return gl_feature

if __name__ == '__main__':

    mixer = voxel_mix_net()
    voxel_feature = torch.randn((4, 80, 128,128))
    mix_feature = torch.cat([voxel_feature,voxel_feature,voxel_feature,voxel_feature,voxel_feature,voxel_feature,voxel_feature,voxel_feature,voxel_feature],dim=1)
    out = mixer(mix_feature)
    out = out.type(torch.HalfTensor)
    exit(0)
