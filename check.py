import torch
print(torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
x = torch.randn(1000, 1000, device="mps")
print((x @ x).shape)