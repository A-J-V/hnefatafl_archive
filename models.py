import torch
from torch import nn
from torch.nn import functional as F
import math

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractionBlock(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=3,
                                                  out_channels=embedding_dim,
                                                  kernel_size=(3, 3),
                                                  stride=1,
                                                  padding='same'
                                                  ),
                                        nn.BatchNorm2d(embedding_dim),
                                        nn.GELU(),
                                        nn.Conv2d(in_channels=embedding_dim,
                                                  out_channels=embedding_dim,
                                                  kernel_size=(3, 3),
                                                  stride=1,
                                                  padding='same'
                                                  ),
                                        nn.BatchNorm2d(embedding_dim),
                                        nn.GELU(),
                                        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, n_dims, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(n_dims, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(n_dims)
        self.mlp = nn.Sequential(
            nn.Linear(n_dims, n_dims * 2),
            nn.GELU(),
            nn.Linear(n_dims * 2, n_dims),
        )
        self.norm2 = nn.LayerNorm(n_dims)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(attn_out + x)
        mlp_out = self.mlp(x)
        x = self.norm2(mlp_out + x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, n_dims):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(n_dims * 121, 512),
            nn.GELU(),
            nn.Linear(512, 40 * 11 * 11),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.policy(x)
        x = x.view(-1, 40, 11, 11)
        return x


class ValueHead(nn.Module):
    def __init__(self, n_dims):
        super().__init__()

        self.value = nn.Sequential(
            nn.Linear(n_dims * 121, 100),
            nn.GELU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.value(x).squeeze()
        return x


class PPOViking(nn.Module):
    def __init__(self, n_dims, n_heads) -> None:
        super().__init__()

        # Encode cell positions
        board_size = 11
        period_scale = 2 * math.pi / board_size
        position_tensor = torch.zeros((2, board_size, board_size))

        for i in range(position_tensor.shape[-1]):
            for j in range(position_tensor.shape[-2]):
                position_tensor[0, i, j] = i
                position_tensor[1, i, j] = j
        position_tensor = position_tensor / 10
        self.position_tensor = position_tensor.unsqueeze(0)

        # Define layers
        self.feature_extraction = FeatureExtractionBlock(n_dims - 3)
        self.attention = nn.Sequential(
            AttentionBlock(n_dims, n_heads),
            AttentionBlock(n_dims, n_heads),
            AttentionBlock(n_dims, n_heads),
        )
        self.policy_head = PolicyHead(n_dims)
        self.value_head = ValueHead(n_dims)

    # Include player into x so that forward doesn't require two args?
    def forward(self, x, player_tensor):
        x = self.feature_extraction(x)

        batch_position = self.position_tensor.to(x.device.type).expand(x.shape[0], 2, 11, 11)
        x = torch.cat((batch_position, x), dim=1)
        x = torch.cat((player_tensor, x), dim=1)

        # Reshape the tensor from [batch, features, h, w] to [batch, h*w, features]
        x = x.view(x.shape[0], x.shape[1], x.shape[2] ** 2)
        x = x.permute(0, 2, 1)

        x = self.attention(x)

        policy_out = self.policy_head(x)
        value_out = self.value_head(x)

        return policy_out, value_out


def load_ai():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using torch version {torch.__version__}.")
    print(f"Using {device}.")

    model = torch.load('./ai_models/PPOViking_v0_2.pth',
                       map_location=torch.device(device))
    return model
