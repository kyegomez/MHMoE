[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Multi-Head Mixture of Experts (MHMoE)

MH-MoE to collectively attend to information from various representation
spaces within different experts to deepen context understanding while significantly enhancing expert activation. 

## install
`pip3 install mh-moe`


## usage
```python
import torch
from mh_moe.main import MHMoE

# Define model parameters
dim = 512
heads = 8
num_experts = 4
num_layers = 3

# Create MHMoE model instance
model = MHMoE(dim, heads, num_experts, num_layers)

# Generate dummy input
batch_size = 10
seq_length = 20
dummy_input = torch.rand(batch_size, seq_length, dim)
dummy_mask = torch.ones(batch_size, seq_length)  # Example mask

# Forward pass through the model
output = model(dummy_input, dummy_mask)

# Print output and its shape
print(output)
print(output.shape)
```