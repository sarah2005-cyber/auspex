import torch.nn
# ---- Embedding rate estimator (simple linear regression head) ----
class EmbeddingRateEstimator(torch.nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, 1)  # predicts embedding rate (0-1)
    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # output between 0 and 1

embed_estimator = EmbeddingRateEstimator()
