import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFeatureFusionSPM(nn.Module):
    def __init__(self, spec_dim=48, fused_dim=64):
        super().__init__()
        # Attention MLP (48 -> 96 -> 48 -> 1)
        self.att_mlp = nn.Sequential(
            nn.Linear(spec_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1)  # scalar weight α_spec
        )

        # Projection to unified 64-dim space
        self.proj = nn.Linear(spec_dim, fused_dim)

    def forward(self, spec_features):
        """
        spec_features: (batch, 48)
        """
        # L2 normalize inputs
        spec_norm = F.normalize(spec_features, p=2, dim=1)  # (batch, 48)

        # Attention weight (scalar per sample)
        att_logits = self.att_mlp(spec_norm)        # (batch, 1)
        alpha_spec = torch.sigmoid(att_logits)      # (batch, 1), in [0,1]

        # Weighted features
        fused = alpha_spec * spec_norm              # (batch, 48)

        # Project to fused_dim
        fused = F.relu(self.proj(fused))            # (batch, 64)

        return fused, alpha_spec
