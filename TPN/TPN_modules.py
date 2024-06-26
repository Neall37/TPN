import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from einops import rearrange


class RFAConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # Use 1D pooling and convolution
        self.get_weight = nn.Sequential(
            nn.AvgPool1d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
            nn.Conv1d(in_channels, in_channels * kernel_size, kernel_size=1, groups=in_channels, bias=False)
        )
        self.generate_feature = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * kernel_size, kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels * kernel_size),
            nn.ReLU()
        )
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        b, c, _ = x.shape
        weight = self.get_weight(x)
        l = weight.shape[2] 
        weighted = weight.view(b, c, self.kernel_size, l).softmax(2)  # Apply softmax across kernel dimension
        feature = self.generate_feature(x).view(b, c, self.kernel_size, l)
        weighted_data = feature * weighted

        # Correctly rearrange for sequence data, considering it's 1D
        conv_data = rearrange(weighted_data, 'b c k l -> b c (k l)')
        output = self.conv(conv_data)
        # output = checkpoint(self.conv, conv_data, use_reentrant=False)
        return output


class ConvBlock(nn.Module):
    # Output Channel: the number of filters you intend to use
    # width: The kernel size for the convolution.
    def __init__(self, in_channels, out_channels, width=1, dilation=1, w_init=None, **kwargs):
        super(ConvBlock, self).__init__()
        # Adjust padding to account for dilation
        padding = (dilation * (width - 1)) // 2
        self.batch_norm = nn.BatchNorm1d(num_features=in_channels)
        # Adding dilation as a parameter to Conv1d
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=width, padding=padding, dilation=dilation, **kwargs)
        self.gelu = nn.GELU()
        # Weight initialization
        if w_init is not None:
            self.conv.weight.data = w_init(self.conv.weight.data.size())

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.gelu(x) 
        x = self.conv(x)
        return x


class StemBlock(nn.Module):
    def __init__(self, in_channels, channels, pooling_type):
        super(StemBlock, self).__init__()
        self.conv1d = ConvBlock(in_channels=in_channels, out_channels=channels // 2, width=3)
        self.pooling = pooling_module(pooling_type, pool_size=2)  # Placeholder for the pooling module

    def forward(self, x):
        x = self.conv1d(x)
        x = self.pooling(x)
        return x


class ConvStack(nn.Module):
    def __init__(self, channels, pooling_type, stack_num=6):
        super(ConvStack, self).__init__()
        self.layers = nn.ModuleList()
        filter_list = exponential_linspace_int(start=channels // 2, end=channels,
                                               num=stack_num, divisible_by=8, in_channels=channels // 2)
        for i in range(len(filter_list) - 1):
            block = nn.Sequential(
                RFAConv1D(in_channels=filter_list[i], out_channels=filter_list[i + 1], kernel_size=3),
                pooling_module(pooling_type, pool_size=2)  # Placeholder for the pooling module
            )
            self.layers.append(block)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate, dropout_rate):
        super(ResidualModule, self).__init__()
        self.main_path = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels,
                      width=3, dilation=dilation_rate),
            nn.Dropout(p=dropout_rate)
        )
        self.shortcut = nn.Identity()  # For the residual connection

    def forward(self, x):
        return self.main_path(x) + self.shortcut(x)


class DilatedConvStack(nn.Module):
    def __init__(self, channels, multi_dilated: float = 1.5,
                 dilation_rate: int = 1, layers_num: int = 11, dropout_rate=0.3):
        super(DilatedConvStack, self).__init__()
        self.layers = nn.ModuleList()

        current_dilation_rate = dilation_rate
        for _ in range(layers_num):
            self.layers.append(ResidualModule(
                in_channels=channels,
                out_channels=channels,
                dilation_rate=current_dilation_rate,
                dropout_rate=dropout_rate
            ))
            current_dilation_rate = int(multi_dilated * current_dilation_rate)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Output(nn.Module):
    def __init__(self, channels, dropout_rate=0.3):
        super(Output, self).__init__()
        self.conv1 = nn.Sequential(
            ConvBlock(in_channels=channels, out_channels=3, width=1),
            nn.Dropout(p=dropout_rate),  # Placeholder for the pooling module
        )
        self.conv2 = nn.Sequential(
            ConvBlock(in_channels=channels, out_channels=5, width=1),
            nn.Dropout(p=dropout_rate),  # Placeholder for the pooling module
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = torch.mean(x1, dim=2)
        x2 = self.conv2(x)
        x2 = torch.mean(x2, dim=2)

        return x1, x2


class TraitProtNet(nn.Module):
    # The main body of the neural network
    def __init__(self, channels=1536, in_channels=480, stack_layer_num=6,
                 layers_num=11, pooling_type='attention', target_percentage=0.8):
        super(TraitProtNet, self).__init__()
        self.stem = StemBlock(in_channels, channels,
                              pooling_type)  # Define pooling_module according to your requirements
        self.conv_stack = ConvStack(channels, pooling_type, stack_num=stack_layer_num)
        self.dilated_conv = DilatedConvStack(channels, multi_dilated=1.5,
                                             dilation_rate=1, layers_num=layers_num,
                                             dropout_rate=0.3)
        self.crop = TargetLengthCrop1D(target_percentage=target_percentage)
        self.head = Output(channels)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv_stack(x)
        x = self.dilated_conv(x)
        x = self.crop(x)
        out_task_1, out_task_2 = self.head(x)
        return out_task_1, out_task_2


class Residual(nn.Module):
    """Residual block."""

    def __init__(self, module, name='residual'):
        super(Residual, self).__init__()
        self._module = module
        self.name = name

    def forward(self, inputs, *args, **kwargs):
        return inputs + self._module(inputs, *args, **kwargs)



def pooling_module(kind, pool_size):
    """Pooling module wrapper."""
    if kind == 'attention':
        return SoftmaxPooling1D(pool_size=pool_size)
    elif kind == 'max':
        return MaxPool1d(kernel_size=pool_size)
    else:
        raise ValueError(f'Invalid pooling kind: {kind}.')


class MaxPool1d(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPool1d, self).__init__()
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=2)

    def forward(self, x):
        x = self.max_pool(x)
        return x


class SoftmaxPooling1D(nn.Module):
    def __init__(self, pool_size: int = 2, per_channel: bool = False,
                 w_init_scale: float = 0.0, name: str = 'global_softmax_pooling'):
        super(SoftmaxPooling1D, self).__init__()
        self.pool_size = pool_size
        self.per_channel = per_channel
        self.w_init_scale = w_init_scale
        self.name = name
        self.logit_linear = None

    def forward(self, inputs):
        if self.logit_linear is None:
            num_features = inputs.shape[-2]
            out_features = num_features if self.per_channel else 1
            self.logit_linear = nn.Linear(num_features, out_features, bias=False)
            if self.w_init_scale == 0.0:
                nn.init.constant_(self.logit_linear.weight, 0)
            else:
                nn.init.normal_(self.logit_linear.weight, mean=0, std=self.w_init_scale)
            self.logit_linear.to(inputs.device)
        batch_size, num_features, length = inputs.shape
        pad_length = (self.pool_size - (length % self.pool_size)) % self.pool_size
        inputs = F.pad(inputs, (0, pad_length), mode='constant', value=0)

        logits = self.logit_linear(inputs.transpose(1, 2))
        weights = F.softmax(logits, dim=1)
        new_length = (length + pad_length) // self.pool_size

        # Reshape inputs and weights to group elements by pool size
        inputs_reshaped = inputs.view(batch_size, new_length, self.pool_size, num_features)
        weights_reshaped = weights.view(batch_size, new_length, self.pool_size, -1)

        pooled = torch.sum(inputs_reshaped * weights_reshaped, dim=2)
        pooled_corrected = pooled.transpose(1, 2)

        return pooled_corrected


    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.logit_linear is not None:
            self.logit_linear.to(*args, **kwargs)
        return self



class TargetLengthCrop1D(nn.Module):
    """Crop or pad sequence to match the desired target length."""

    def __init__(self, target_percentage):
        super(TargetLengthCrop1D, self).__init__()
        self.target_percentage = target_percentage

    def forward(self, inputs):
        current_length = inputs.shape[-1]
        dim = len(inputs.shape)
        target_length = int(current_length * self.target_percentage)
        # If the sequence is longer than the target, crop it.
        if dim > 2:
            trim = (current_length - target_length) // 2
            return inputs[:, :, trim:trim + target_length]
        elif dim == 2:
            trim = (current_length - target_length) // 2
            return inputs[:, trim:trim + target_length]


def exponential_linspace_int(start, end, num, divisible_by=1, in_channels=None):
    """Generate a list where the first number is X, followed by exponentially increasing integers from start to end.
    Each number after X in the sequence is made divisible by 'divisible_by'."""

    def _round(x):
        # Round x to be divisible by divisible_by
        return int(round(x / divisible_by) * divisible_by)

    if num < 2:
        raise ValueError("num must be at least 2 to include X and one more value.")

    # Adjust the base calculation to fit the reduced number for the exponential sequence by one
    base = np.exp(np.log(end / start) / (num - 2))

    # Initialize the sequence with X as the first element
    sequence = [in_channels] if in_channels is not None else [_round(start)]

    # Generate the rest of the sequence
    for i in range(1, num):
        next_value = _round(start * base ** (i - 1))
        sequence.append(next_value)

    return sequence



