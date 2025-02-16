import torch
import torch.nn as nn
class BTModel(nn.Module):
    def __init__(self, N):
        super(BTModel, self).__init__()
        self.reward = nn.Parameter(torch.ones(N))

    def forward_exp(self, chosen_id, rejected_id):
        reward_chosen = torch.exp(self.reward[chosen_id])
        reward_rejected = torch.exp(self.reward[rejected_id])
        return reward_chosen / (reward_chosen + reward_rejected)

    def forward_sigmoid(self, chosen_id, rejected_id):
        reward_chosen = self.reward[chosen_id]
        reward_rejected = self.reward[rejected_id]
        return torch.sigmoid(reward_chosen - reward_rejected)

    def loss(self, pred, label):
        return -torch.log(pred) if label == 1 else -torch.log(1 - pred)