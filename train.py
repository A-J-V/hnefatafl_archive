import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import random
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

from models import load_ai


class HnefataflDataset(Dataset):
    def __init__(self, data_path, player=None):

        df_list = []
        print("Loading dataset...")
        for file in tqdm(os.listdir(data_path)):
            record_path = os.path.join(data_path, file)
            df_list.append(
                pd.read_csv(
                    record_path,
                    on_bad_lines='skip'))

        self.data = pd.concat(df_list, ignore_index=True)
        if player == 1:
            print(f"Filtering data to player 1.")
            self.data = self.data.loc[self.data['turn'] == 1]
        elif player == 0:
            print(f"Filtering data to player 0.")
            self.data = self.data.loc[self.data['turn'] == 0]
        print(f"Data contains the following player: {self.data['turn'].unique()}")
        self.data = self.data.to_numpy(dtype=np.float32, copy=True)
        self.player = player

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        game_state = self.data[idx][: 121]
        game_state = self.reconstruct_board(game_state)

        action_selected, action_prob = self.data[idx][121: 123]

        game_action_mask = self.data[idx][123: -8]

        winner = self.data[idx][-8]

        player = self.get_player_tensor()

        gae_attacker = self.data[idx][-2]
        gae_defender = self.data[idx][-1]

        return (game_state,
                torch.tensor(action_selected).long(),
                torch.tensor(action_prob).float(),
                torch.tensor(winner).float(),
                torch.tensor(game_action_mask).float(),
                player,
                torch.tensor(gae_attacker).float(),
                torch.tensor(gae_defender).float(),
                )

    def reconstruct_board(self, flat_board: np.ndarray) -> torch.tensor:
        reshaped = flat_board.reshape(11, 11).astype('int')
        reconstructed = np.zeros((3, 11, 11))
        rows, cols = np.indices((11, 11))
        valid_indices = (reshaped != -1).astype('int')
        reconstructed[reshaped[valid_indices], rows[valid_indices], cols[valid_indices]] = 1
        return torch.from_numpy(reconstructed).float()

    def get_action_target(self, action: np.ndarray) -> torch.tensor:
        move, row, col = action
        action_target = torch.zeros((40, 11, 11))
        action_target[move, row, col] = 1
        return action_target.float()

    def get_player_tensor(self):
        return torch.zeros(1, 11, 11).float() + self.player


class PPOLoss(nn.Module):
    def __init__(self, e=0.2, c1=0.5, c2=None):
        super().__init__()
        self.e = e
        self.c1 = c1
        self.c2 = c2
        self.vf_loss = nn.MSELoss()

    def forward(self,
                policy_pred,
                value_pred,
                op_action,
                op_prob,
                winner,
                player,
                gae_a,
                gae_d,
                ):
        """
        :policy_pred: A tensor of probabilities over the (flattened) action space.
        :value_pred: A tensor of probabilities that the defenders win.
        :op_action: A tensor of old policy actions selected.
        :op_prob: A tensor of old policy probabilities for that action.
        :winner: A binary tensor of who won this game, 1 for attackers, 0 for defenders.
        :player: A binary tensor of whose turn it is, 1 for attackers, 0 for defenders.
        """
        rewards_to_go = torch.where(player == winner, 1, -1).float()

        if torch.isnan(policy_pred).any():
            print("########################################")
            print("NAN IN POLICY PREDICTIONS IN LOSS FUNCTION!")
            print("########################################")
        if torch.isnan(value_pred).any():
            print("########################################")
            print("NAN IN VALUE PREDICTIONS IN LOSS FUNCTION!")
            print("########################################")
        # GAE depends on player perspective, so make sure that we are using GAE
        # with respect to the player whose turn it is in each sample
        advantage = torch.where(player == 1, gae_a, gae_d)

        # Calculate l_clip using the ratio and advantage
        np_prob = torch.gather(policy_pred, 1, op_action.unsqueeze(-1)).squeeze()
        ratio = (np_prob / op_prob) * advantage
        clipped_ratio = torch.clamp(ratio, 1 - self.e, 1 + self.e) * advantage
        l_clip = torch.min(ratio, clipped_ratio)
        l_clip = -l_clip.mean()

        # We're using value propagation since there is only one reward at the end of the game.
        # Therefore, true value = rewards_to_go
        l_vf = self.vf_loss(value_pred, rewards_to_go)

        # Combine the loss and if c2 was passed, include an entropy bonus
        if self.c2 is not None:
            # Calculating the entropy
            entropy = -(policy_pred * (policy_pred + 0.0000001).log()).sum(-1)
            entropy = entropy.mean()
            loss = l_clip + self.c1 * l_vf - self.c2 * entropy
        else:
            loss = l_clip + self.c1 * l_vf

        return loss


def train_ppo(model, loss_fn, device, dataloader, optimizer, epoch):
    model.train()
    attacker_winrate = []
    print("Epoch: ", epoch)
    for batch_idx, (data, op_action, op_prob, winner, mask, player, gae_a, gae_d) in enumerate(dataloader):
        data, op_action, op_prob, winner, mask, player, gae_a, gae_d = (data.to(device),
                                                                        op_action.to(device),
                                                                        op_prob.to(device),
                                                                        winner.to(device),
                                                                        mask.to(device),
                                                                        player.to(device),
                                                                        gae_a.to(device),
                                                                        gae_d.to(device),
                                                                        )
        optimizer.zero_grad()

        policy_probs, value_probs = model.pred_probs(data, player, mask)

        loss = loss_fn(policy_probs,
                       value_probs,
                       op_action,
                       op_prob,
                       winner,
                       player[:, 0, 0, 0],
                       gae_a,
                       gae_d)

        attacker_winrate.append(winner.mean())

        loss.backward()

        optimizer.step()

        if batch_idx % 5 == 0:
            print("Training set [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))


def training_iteration(model,
                       player,
                       batch_size=4096,
                       learning_rate=0.00001,
                       weight_decay=0.0,
                       epochs=2,
                       c1=0.5,
                       c2=0.05,
                       checkpoint=None
                       ):

    print(f"Loading model '{model}'")
    m = load_ai(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)
    optimizer = torch.optim.Adam(m.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    loss_fn = PPOLoss(e=0.2, c1=c1, c2=c2)
    dataset = HnefataflDataset(data_path="./game_recordings",
                               player=player,
                               )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        train_ppo(m, loss_fn, device, dataloader, optimizer, epoch)
    torch.save(m, f"./ai_models/{model}.pth")
    if checkpoint is not None:
        torch.save(m.state_dict(), f"./checkpoints/{model}_1_{checkpoint}.pth")
