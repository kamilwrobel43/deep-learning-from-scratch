import torch
import torch.nn as nn


"""
Vanilla UNet implementation from paper: U-Net: Convolutional Networks for Biomedical Image Segmentation (2015) - https://arxiv.org/abs/1505.04597
"""

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3,3), 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3,3), 1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d((2,2), stride = 2)


    def forward(self, x):
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x, skip
    

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None, final_channels: int | None = None):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3,3), 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3,3), 1),
            nn.ReLU()
        )
        if not final_channels:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(out_channels, out_channels // 2, (2,2), 1, padding="same")
            )
        else:
            self.upsample = nn.Conv2d(out_channels, final_channels, (1,1))


    def _crop_and_cat(self, x, skip: torch.Tensor):

        B, C, H_x, W_x = x.shape
        _, _, H_s, W_s = skip.shape

        H_diff = (H_s - H_x) // 2
        W_diff = (W_s - W_x) // 2


        skip_cropped = skip[:, :, H_diff: H_diff+H_x, W_diff: W_diff + W_x]
        x = torch.cat((x, skip_cropped), dim = 1)
        
        return x

    def forward(self, x, skip):
        x = self._crop_and_cat(x, skip)
        x = self.conv(x)
        x = self.upsample(x)

        return x
    

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3,3), 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3,3), 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(out_channels, out_channels // 2, (2,2), 1, padding="same")
        )

    def forward(self, x):
        return self.network(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, n_blocks: int ):
        super().__init__()

        down_blocks = [DownsamplingBlock(in_channels, base_channels)]
        
        curr_in = base_channels
        for _ in range(n_blocks-1):
            curr_out = curr_in*2
            down_blocks.append(DownsamplingBlock(curr_in, curr_out))
            curr_in = curr_out
        
        
        self.down_blocks = nn.ModuleList(down_blocks)

        curr_out = curr_in*2
        self.bottleneck_block = BottleneckBlock(curr_in, curr_in*2)
        curr_in = curr_out

        up_blocks = []
        for _ in range(n_blocks-1):
            curr_out = curr_in // 2
            up_blocks.append(UpsamplingBlock(curr_in, curr_out))
            curr_in = curr_out

        up_blocks.append(UpsamplingBlock(curr_in, curr_in // 2, 2))
        self.up_blocks = nn.ModuleList(up_blocks)

    def forward(self, x):
        skips = []
        for block in self.down_blocks:
            x, skip = block(x)
            skips.append(skip)

        x = self.bottleneck_block(x)

        for block in self.up_blocks:
            x = block(x, skips.pop())
        
        return x


if __name__ == "__main__":
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    x = torch.empty(2, 1, 572, 572).to(device)

    unet = UNet(in_channels=1, base_channels=64, n_blocks=4).to(device)
    for _ in range(100):
        out = unet(x)

    print(out.shape)

    # in_channels = 512
    # out_channels = in_channels*2
    # network = nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, (3,3), 1),
    #         nn.Conv2d(out_channels, out_channels, (3,3), 1),
    #         nn.Upsample(scale_factor=2),
    #         nn.Conv2d(out_channels, out_channels // 2, (2,2), 1, padding="same")
    #     )

    # x = torch.empty(32, 512, 32, 32)

    # out = network(x)
    # print(out.shape)


    # # x_up_in = torch.empty(32, 64, 392, 392)
    # # ds_block = DownsamplingBlock(1, 64)
    # # up_block = UpsamplingBlock(128, 64, 2)

    # out, skip = ds_block(x)

    # out_up  = up_block(x_up_in, skip)

    # print(out.shape)
    # print(skip.shape)
    # print(out_up.shape)

