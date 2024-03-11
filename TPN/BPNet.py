import torch
import torch.nn as nn
import torch.nn.functional as F


class BPNet(nn.Module):
    def __init__(self, num_chars=480, num_dilated_conv=9):
        super(BPNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_chars, out_channels=64, kernel_size=25, padding=12)

        self.dilated_conv = nn.ModuleList([
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2 ** i, dilation=2 ** i)
            for i in range(num_dilated_conv)
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.task1_fc = nn.Linear(64, 3)
        self.task2_fc = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        for conv in self.dilated_conv:
            conv_x = F.relu(conv(x))
            x = conv_x + x

        # Use global average pooling to handle variable-length inputs
        x = self.global_avg_pool(x).view(x.size(0), -1)
        x1 = self.task1_fc(x)

        x2 = self.task2_fc(x)

        return x1, x2
