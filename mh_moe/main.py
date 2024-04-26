import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from zeta.nn.modules import NormalSparseMoE
from einops import reduce, rearrange


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, dim: int, head, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(dim, head, bias=False)
        self.query = nn.Linear(dim, head, bias=False)
        self.value = nn.Linear(dim, head, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


# Changing the above to accomodate noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        # layer for router logits
        self.topkroute_linear = nn.Linear(dim, num_experts)
        self.noise_linear = nn.Linear(dim, num_experts)

    def forward(self, mh_output):
        print(mh_output.shape)
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        print(f"Router output shape: {router_output.shape}")
        return router_output, indices


# Expert module
class Expert(nn.Module):
    """An MLP is a simple linear layer followed by a non-linearity i.e. each Expert"""

    def __init__(self, dim, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SparseMoE(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(dim, num_experts, top_k)
        self.experts = nn.ModuleList(
            [Expert(dim) for _ in range(num_experts)]
        )
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(
            -1, gating_output.size(-1)
        )

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[
                    flat_mask, i
                ].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(
                    1
                )

        return final_output


# First define the top k router module
class TopkRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(dim, num_experts)

    def forward(self, x):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.linear(x)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        print(router_output.shape)
        return router_output, indices


class MHMoE(nn.Module):
    def __init__(self, dim, num_heads, num_experts, num_layers):
        super(MHMoE, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.num_layers = num_layers

        # Create a module list for multi-head layers and merge layers
        self.multi_head_layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_layers)]
        )
        self.merge_layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_layers)]
        )

        # Initialize parameters
        for i in range(num_layers):
            nn.init.xavier_uniform_(
                self.multi_head_layers[i].weight,
                gain=1 / math.sqrt(2),
            )
            nn.init.xavier_uniform_(self.merge_layers[i].weight)
            nn.init.constant_(self.merge_layers[i].bias, 0)

    def forward(self, x, mask):
        # Loop through each layer
        for i in range(self.num_layers):
            x = self.process_layer(x, mask, i)
        return x

    # def process_layer(self, x, mask, layer_index):
    #     batch_size, length, _ = x.size()

    #     # Processed by multi-head layer
    #     x = self.multi_head_layers[layer_index](x)
    #     print(x.shape)

    #     # Using einops to split and rearrange sub-tokens in parallel
    #     x = rearrange(
    #         x,
    #         "b l (h d) -> (b h) l d",
    #         h=self.num_heads,
    #         d=self.dim // self.num_heads,
    #     )
    #     b, s, d = x.shape
    #     print(x.shape)

    #     # Example routing logic (placeholder)
    #     # x, i = NoisyTopkRouter(self.dim, self.num_experts, 2)(x)  # Replace with actual routing logic
    #     x, i = TopkRouter(d, self.num_experts, 2)(x)
    #     print(x.shape)

    #     # Sparse Moe
    #     # x, e = NormalSparseMoE(
    #     #     dim,
    #     #     num_experts=self.num_experts,
    #     #     # experts=
    #     # )

    #     # Using einops to merge back to the original token form
    #     x = rearrange(
    #         x,
    #         "(b h) l d -> b l (h d)",
    #         b=batch_size,
    #         h=self.num_heads,
    #         d=self.dim // self.num_heads,
    #     )

    #     # Output processed by merge layer
    #     x = self.merge_layers[layer_index](x)

    #     return x

    def process_layer(self, x, mask, layer_index):
        batch_size, length, _ = x.size()

        # Processed by multi-head layer
        x = self.multi_head_layers[layer_index](x)

        # Correcting the reshaping step
        # We need to ensure x is reshaped to (batch_size, num_heads, length, dim/num_heads)
        x = x.view(
            batch_size,
            length,
            self.num_heads,
            self.dim // self.num_heads,
        )
        x = x.permute(
            0, 2, 1, 3
        ).contiguous()  # this rearranges to (batch_size, num_heads, length, dim/num_heads)
        x = x.view(
            batch_size * self.num_heads,
            length,
            self.dim // self.num_heads,
        )
        b, s, d = x.shape
        print(x.shape)

        # Simulated expert processing (needs actual implementation)
        # For now, assume identity transformation
        # x = x  # Replace with actual routing and processing logic
        # x, i = NoisyTopkRouter(d, self.num_experts, 2)(x)
        # print(x.shape)
        # x, e = NormalSparseMoE(
        #     self.dim,
        #     self.num_experts,
        # )
        # x = TopkRouter(
        #     d,
        #     self.num_experts,
        #     top_k=4,
        # )(x)
        # x = reduce("b h l d -> b l (h d)", x, "mean")
        x = SparseMoE(d, self.num_experts, 2)(x)

        # Reshape back to original form after processing
        x = x.view(
            batch_size,
            self.num_heads,
            length,
            self.dim // self.num_heads,
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, length, self.dim)

        # Output processed by merge layer
        x = self.merge_layers[layer_index](x)

        return x
