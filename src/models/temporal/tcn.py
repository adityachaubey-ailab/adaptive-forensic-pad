import torch
import torch.nn as nn
from typing import Optional

class TemporalBlock(nn.Module):
    """
    Temporal Block for TCN. Uses causal convolution to prevent future information leakage.
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()
        # Causal convolution
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        # Ensure causality by trimming the end padding if needed
        self.chomp1 = nn.ConstantPad1d((-padding, 0), 0)
        
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = nn.ConstantPad1d((-padding, 0), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network.
    Expects input features over time and summarizes them temporally.
    """
    def __init__(
        self,
        input_dim: int,
        num_channels: list[int] = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, seq_len, feature_dim) sequence of per-frame features
        Returns:
            shape (B, temporal_feature_dim) temporal summary
        """
        # TCN expects (B, C, L)
        x = x.transpose(1, 2)
        out = self.network(x)
        # Take the last time step as the temporal summary
        out = out[:, :, -1]
        return out
