import torch
from torch import nn


class ResNet_initial_block(nn.Module):
    """
    Accept (batch_size, 3, 224, 224) shaped tensor and pass it through
    a 7x7 convolution and MaxPool layer.
    """

    def __init__(self):
        super().__init__()

        # The convolution, batchnorm, and ReLU activation.
        # Remember to add the padding!
        self.resnet_block1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                                     out_channels=64,
                                                     kernel_size=(7, 7),
                                                     stride=1,
                                                     padding='same'
                                                     ),
                                           nn.BatchNorm2d(64),
                                           nn.ReLU(),
                                           )

        # The Max Pool layer. Remember to add the padding!
        # self.pool = nn.MaxPool2d(kernel_size=3,
        #                         stride=2,
        #                         padding=1,
        #                        )

    def forward(self, x):
        x = self.resnet_block1(x)
        # x = self.pool(x)
        return x


class ResNet_basic_block(nn.Module):
    """Accept tensors and do these operations: conv, relu, conv, add residual, relu."""

    def __init__(self,
                 channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=channels,
                                             out_channels=channels,
                                             kernel_size=(3, 3),
                                             padding='same'
                                             ),
                                   nn.BatchNorm2d(channels),
                                   nn.ReLU()
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=channels,
                                             out_channels=channels,
                                             kernel_size=(3, 3),
                                             padding='same'
                                             ),
                                   nn.BatchNorm2d(channels)
                                   )

        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += res
        return self.relu(x)


class ResNet_transition_block(nn.Module):
    """Accept tensors and do these operations: conv, relu, conv, add residual, relu."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=(3, 3),
                                             stride=2,
                                             padding=1
                                             ),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU()
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
                                  stride=2
                                  )

        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += res
        return self.relu(x)


class ResNet(nn.Module):
    """Model architecture based on the the paper 'Deep Residual Learning for Image Recognition'
       by He, Zhang, Ren, and Sun (2015).

       This is the 34-layer version.
       """

    def __init__(self, classes=1) -> None:
        super().__init__()

        self.resnet = nn.Sequential(*[
            ResNet_initial_block(),
            nn.Sequential(*(ResNet_basic_block(64) for _ in range(3))),
            ResNet_transition_block(64, 128),
            nn.Sequential(*(ResNet_basic_block(128) for _ in range(3))),
            ResNet_transition_block(128, 256),
            nn.Sequential(*(ResNet_basic_block(256) for _ in range(4))),
            ResNet_transition_block(256, 512),
            nn.Sequential(*(ResNet_basic_block(512) for _ in range(2))),
            # nn.AvgPool2d(kernel_size=(7, 7)),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=classes),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        return self.resnet(x)


def load_baby_viking():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using torch version {torch.__version__}.")
    print(f"Using {device}.")

    model = torch.load('./ai_models/BabyViking.pth',
                       map_location=torch.device(device))
    return model
