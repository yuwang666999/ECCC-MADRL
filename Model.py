import torch
import torch.nn as nn
import torch.nn.functional as F



# 定义 SWA 模块
class SWA(nn.Module):
    def __init__(self, in_channels, n_heads=8, window_size=7):
        super(SWA, self).__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.window_size = window_size

        self.query_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, length = x.size()
        padded_x = F.pad(x, [self.window_size // 2, self.window_size // 2], mode='constant', value=0)

        proj_query = self.query_conv(x).view(batch_size, self.n_heads, C // self.n_heads, length)
        proj_key = self.key_conv(padded_x).unfold(2, self.window_size, 1)
        proj_key = proj_key.permute(0, 1, 3, 2).contiguous().view(batch_size, self.n_heads, C // self.n_heads, -1)
        proj_value = self.value_conv(padded_x).unfold(2, self.window_size, 1)
        proj_value = proj_value.permute(0, 1, 3, 2).contiguous().view(batch_size, self.n_heads, C // self.n_heads, -1)

        energy = torch.matmul(proj_query.permute(0, 1, 3, 2), proj_key)
        attention = self.softmax(energy)

        out_window = torch.matmul(attention, proj_value.permute(0, 1, 3, 2))
        out_window = out_window.permute(0, 1, 3, 2).contiguous().view(batch_size, C, length)

        out = self.gamma * out_window + x
        return out

    # 定义 ActorNetwork

# 定义 DSC 模块
class DSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out

# 定义 IDSC 模块
class IDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out

# 定义 ActorNetwork
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, output_activation, init_w=3e-3):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.swa = SWA(in_channels=64, n_heads=8, window_size=7)  # Insert SWA here

        # Insert DSC and IDSC here
        self.dsc = DSC(c_in=64, c_out=64, k_size=3, stride=1, padding=1)
        self.idsc = IDSC(c_in=64, c_out=64, k_size=3, stride=1, padding=1)

        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        self.output_activation = output_activation

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = out.view(out.size(0), 64, 1, 1)  # Reshape to (batch_size, channels, height, width)

        # Apply SWA
        out = self.swa(out.squeeze(-1)).unsqueeze(-1)  # Adjust dimensions for SWA

        # Apply DSC and IDSC
        out = self.dsc(out)
        out = self.idsc(out)

        out = out.view(out.size(0), -1)  # Flatten back to (batch_size, features)
        out = F.relu(self.fc2(out))
        out = self.output_activation(self.fc3(out))
        return out

# 定义 CriticNetwork
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, pestate, peraction, output_size=1, init_w=3e-3):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim + pestate + peraction, 512)
        self.swa = SWA(in_channels=512, n_heads=8, window_size=7)  # Insert SWA here

        # # Insert DSC and IDSC here
        self.dsc = DSC(c_in=512, c_out=512, k_size=3, stride=1, padding=1)
        self.idsc = IDSC(c_in=512, c_out=512, k_size=3, stride=1, padding=1)

        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, pstate, paction):
        # Ensure all inputs are 2D tensors
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        if len(pstate.shape) == 1:
            pstate = pstate.unsqueeze(0)
        if len(paction.shape) == 1:
            paction = paction.unsqueeze(0)

        out = torch.cat([state, action, pstate, paction], dim=1)
        out = F.relu(self.fc1(out))
        out = out.view(out.size(0), 512, 1, 1)  # Reshape to (batch_size, channels, height, width)

        # Apply SWA
        out = self.swa(out.squeeze(-1)).unsqueeze(-1)  # Adjust dimensions for SWA

        # # Apply DSC and IDSC
        # out = self.dsc(out)
        out = self.idsc(out)

        out = out.view(out.size(0), -1)  # Flatten back to (batch_size, features)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out