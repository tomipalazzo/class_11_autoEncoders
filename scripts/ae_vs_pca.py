# Linear Autoencoder vs PCA
import torch
import torch.nn as nn

torch.manual_seed(0)

# 1) Data (mean-centered)
n, d, k = 2000, 20, 5
U = torch.linalg.qr(torch.randn(d, k)).Q
Z = torch.randn(n, k) * torch.linspace(3.0, 1.0, k)
X = Z @ U.T + 0.95 * torch.randn(n, d)
X -= X.mean(0, keepdim=True)

# 2) PCA projection
_, _, Vh = torch.linalg.svd(X, full_matrices=False)
V_k = Vh.T[:, :k]
X_pca = X @ V_k @ V_k.T


# 3) Linear AE
class LinearAE(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.encoder = nn.Linear(d, k, bias=False)
        self.decoder = nn.Linear(k, d, bias=False)
        nn.init.normal_(self.encoder.weight, std=0.05)
        nn.init.zeros_(self.decoder.weight)

    def forward(self, x):
        # identity activation, tie W_d = W_e^T each forward
        self.decoder.weight.data = self.encoder.weight.data.t()
        return self.decoder(self.encoder(x))


ae = LinearAE(d, k)
opt = torch.optim.Adam(ae.parameters(), lr=0.02)
loss_fn = nn.MSELoss()

# 4) Train
for _ in range(1000):
    X_hat = ae(X)
    loss = loss_fn(X_hat, X)
    opt.zero_grad()
    loss.backward()
    opt.step()

# 5) Compare only projections (no angles/cosines)
with torch.no_grad():
    W_e = ae.encoder.weight.t()
    X_ae = X @ W_e @ W_e.T  # AE projection
    fro_pca = torch.linalg.matrix_norm(X_pca)
    fro_ae = torch.linalg.matrix_norm(X_ae)
    proj_gap = torch.linalg.matrix_norm(X_pca - X_ae)

print("||X_pca||_F:", float(fro_pca))
print("||X_ae ||_F:", float(fro_ae))
print("Projection gap ||X_pca - X_ae||_F:", float(proj_gap))
