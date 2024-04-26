import torch
from mh_moe.main import MHMoE

# Example usage:
dim = 512
num_heads = 8
num_experts = 4
num_layers = 3
model = MHMoE(dim, num_heads, num_experts, num_layers)
batch_size = 10
seq_length = 20
dummy_input = torch.rand(batch_size, seq_length, dim)
dummy_mask = torch.ones(batch_size, seq_length)  # Example mask
output = model(dummy_input, dummy_mask)
print(output)
print(output.shape)
