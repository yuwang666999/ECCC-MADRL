import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# 1. 自窗口注意力
# --------------------------------------------------
class SelfWindowAttention(nn.Module):
    def __init__(self, D_in=7, h=4, w=5, D_h=None):
        super().__init__()
        self.D_in = D_in
        self.h   = h
        self.w   = w
        self.D_h = D_h if D_h is not None else max(1, D_in // h)

        self.W_q = nn.Linear(D_in, self.D_h * h)
        self.W_k = nn.Linear(D_in, self.D_h * h)
        self.W_v = nn.Linear(D_in, self.D_h * h)

        self.out_proj = nn.Linear(self.D_h * h, 8)

    def forward(self, x):
        B, L, _ = x.shape
        padded  = F.pad(x, [0, 0, self.w//2, self.w//2], mode='constant', value=0)  # (B, L+pad, D_in)

        Q = self.W_q(x).view(B, L, self.h, self.D_h).transpose(1, 2)          # (B, h, L, D_h)
        K = self.W_k(padded).view(B, padded.size(1), self.h, self.D_h).transpose(1, 2)
        V = self.W_v(padded).view(B, padded.size(1), self.h, self.D_h).transpose(1, 2)

        # 注意力得分
        score = torch.matmul(Q, K.transpose(-2, -1)) / (self.D_h ** 0.5)      # (B, h, L, L+pad)

        # 构造窗口掩码 (h, L, L+pad)
        window_mask = torch.zeros(self.h, L, padded.size(1), device=x.device)
        for i in range(L):
            start = max(0, i - self.w//2)
            end   = min(padded.size(1), i + self.w//2 + 1)
            window_mask[:, i, start:end] = 1
        score = score.masked_fill(window_mask.unsqueeze(0) == 0, -1e9)

        attn = F.softmax(score, dim=-1)
        out  = torch.matmul(attn, V)                                          # (B, h, L, D_h)
        out  = out.transpose(1, 2).contiguous().view(B, L, self.h * self.D_h) # (B, L, h*D_h)
        return self.out_proj(out)                                             # (B, L, 8)

# --------------------------------------------------
# 2. 深度可分离卷积
# --------------------------------------------------
class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_ch=8, out_ch=8, k=3):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, k, padding=k//2, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# --------------------------------------------------
# 3. 改进的深度可分离卷积（反向 + 残差）
# --------------------------------------------------
class ImprovedDepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_ch=8, out_ch=8, k=3, lambda_res=0.3):
        super().__init__()
        self.lambda_res = lambda_res
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.depthwise = nn.Conv2d(out_ch, out_ch, k, padding=k//2, groups=out_ch)

    def forward(self, x, residual=None):
        out = self.pointwise(x)
        out = self.depthwise(out)
        if residual is not None:
            out = out + self.lambda_res * residual
        return out


# --------------------------------------------------
# 4. ECCC 特征提取器
# --------------------------------------------------
class ECCCFeatureExtractor(nn.Module):
    def __init__(self, D_in=7, h=4, w=5):
        super().__init__()
        self.swa  = SelfWindowAttention(D_in, h, w)
        self.dsc  = DepthwiseSeparableConvolution(8, 8)
        self.idsc = ImprovedDepthwiseSeparableConvolution(8, 8)
        self.final_proj = nn.Linear(8, 8)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)                      # (B, 1, D_in)
        swa_out = self.swa(x)                       # (B, L, 8)
        conv_in = swa_out.transpose(1, 2).unsqueeze(-1)  # (B, 8, L, 1)
        dsc_out  = self.dsc(conv_in)
        # 简易残差：直接复用 conv_in
        idsc_out = self.idsc(dsc_out, conv_in)
        linear_in = idsc_out.squeeze(-1).transpose(1, 2)  # (B, L, 8)
        out = self.final_proj(linear_in).squeeze(1)       # (B, 8)
        return out


# --------------------------------------------------
# 5. Actor 网络
# --------------------------------------------------
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, output_activation=nn.Tanh(), init_w=3e-3):
        super().__init__()
        self.extractor = ECCCFeatureExtractor(state_dim)
        self.head = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-init_w, init_w)
                m.bias.data.uniform_(-init_w, init_w)
        self.act = output_activation

    def forward(self, state):
        feat = self.extractor(state)
        return self.act(self.head(feat))


# --------------------------------------------------
# 6. Critic 网络（Per-action DQN）
# --------------------------------------------------
class CriticNetwork(nn.Module):
    def __init__(self, global_state_dim, global_action_dim,
                 per_state_dim, per_action_dim, output_size=1, init_w=3e-3):
        super().__init__()
        self.global_net = nn.Sequential(
            nn.Linear(global_state_dim + global_action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.per_net = nn.Sequential(
            nn.Linear(per_state_dim + per_action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.q_head = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        for m in [self.global_net, self.per_net, self.q_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.uniform_(-init_w, init_w)
                    layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, global_state, global_action, per_state, per_action):
        g = self.global_net(torch.cat([global_state, global_action], dim=-1))
        p = self.per_net(torch.cat([per_state, per_action], dim=-1))
        return self.q_head(torch.cat([g, p], dim=-1))