import torch
import torch_directml
dml = torch_directml.device()
print(f"Using device: {dml}")
x = torch.tensor([1.0]).to(dml)
print(x)