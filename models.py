import torch
from torch import nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

board_size = 11
period_scale = 2 * math.pi / board_size
position_tensor = torch.zeros((4, board_size, board_size))

# Encode positions
for i in range(position_tensor.shape[-1]):
    for j in range(position_tensor.shape[-2]):
        position_tensor[0, i, j] = math.sin(i * period_scale)
        position_tensor[1, i, j] = math.cos(i * period_scale)
        position_tensor[2, i, j] = math.sin(i * period_scale)
        position_tensor[3, i, j] = math.cos(i * period_scale)
position_tensor = position_tensor.unsqueeze(0).expand(1, 4, board_size, board_size).to(device)


class InitialBlock(nn.Module):
    def __init__(self):
        super().__init__()

        # The convolution, batchnorm, and ReLU activation.
        # Remember to add the padding!
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=3,
                                                  out_channels=64,
                                                  kernel_size=(3, 3),
                                                  stride=1,
                                                  padding='same'
                                                  ),
                                        nn.BatchNorm2d(64),
                                        nn.GELU(),
                                        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class BasicBlock(nn.Module):
    """A Convolutional layer that outputs the spatial and channel dims as are input."""

    def __init__(self,
                 channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=channels,
                                             out_channels=channels,
                                             kernel_size=(3, 3),
                                             padding='same'
                                             ),
                                   nn.BatchNorm2d(channels),
                                   nn.GELU()
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=channels,
                                             out_channels=channels,
                                             kernel_size=(3, 3),
                                             padding='same'
                                             ),
                                   nn.BatchNorm2d(channels)
                                   )

        self.relu = nn.GELU()

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += res
        return self.relu(x)


class TransitionBlock(nn.Module):
    """A Conv layer that outputs more channels as are input but same spatial dims."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int
                 ):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=(3, 3),
                                             stride=1,
                                             padding='same'
                                             ),
                                   nn.BatchNorm2d(out_channels),
                                   nn.GELU()
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels,
                                             out_channels=out_channels,
                                             kernel_size=(3, 3),
                                             padding='same'
                                             ),
                                   nn.BatchNorm2d(out_channels)
                                   )

        #
        self.residual = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(1, 1),
                                  stride=1,
                                  )

        self.relu = nn.GELU()

    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += res
        return self.relu(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_dims, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(n_dims, n_heads)

    def forward(self, x):
        x = torch.cat((position_tensor, x), dim=1)
        # Flatten the input from (batch_size, features, height, width)
        # to (batch_size, features, height*width) and permute it to
        # (height*width, batch_size, features) then send it through attention
        x = x.view(x.shape[0], x.shape[1], x.shape[2] ** 2)
        x = x.permute(2, 0, 1)
        x, _ = self.attention(x, x, x)
        return x


class NeuralViking2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_section = nn.Sequential(*[
            InitialBlock(),
            nn.Sequential(*(BasicBlock(64) for _ in range(3))),
            TransitionBlock(64, 128),
            nn.Sequential(*(BasicBlock(128) for _ in range(3))),
            TransitionBlock(128, 256),
            nn.Sequential(*(BasicBlock(256) for _ in range(3))),
            TransitionBlock(256, 512),
            nn.Sequential(*(BasicBlock(512) for _ in range(2))),
        ])

        self.attention_section = AttentionBlock(n_dims=516,
                                                n_heads=6,
                                                )
        self.classifier = nn.Sequential(
            nn.Linear(516, 516),
            nn.GELU(),
            nn.Linear(516, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.conv_section(x)
        x = self.attention_section(x)
        x = self.classifier(x[0])
        return x


def load_ai():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using torch version {torch.__version__}.")
    print(f"Using {device}.")

    model = torch.load('./ai_models/NeuralViking2.pth',
                       map_location=torch.device(device))
    return model
