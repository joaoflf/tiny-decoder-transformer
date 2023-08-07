from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        hidden_size: int,
        context_size: int,
        vocab_size: int,
    ):
        super().__init__()
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding_table = nn.Embedding(context_size, hidden_size)
        head_size = hidden_size // num_heads
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.blocks = nn.Sequential(
            *[
                Block(num_heads, head_size, hidden_size, context_size)
                for _ in range(num_blocks)
            ]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, x: torch.Tensor | dict, target: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        has_attention_mask = isinstance(x, dict)
        inputs = x["input_ids"] if has_attention_mask else x
        attention_mask = x["attention_mask"] if has_attention_mask else None
        B, T = inputs.shape
        token_embedding = self.token_embedding_table(inputs)
        position_embedding = self.position_embedding_table(
            torch.arange(T, device=inputs.device)
        )
        x = token_embedding + position_embedding
        x = (
            self.blocks(x)
            if attention_mask is None
            else self.blocks((x, attention_mask))[0]
        )
        x = self.layer_norm(x)
        logits = self.lm_head(x)  #

        if target is None:
            loss = None
        else:
            logits = logits.view(B * T, self.vocab_size)
            loss = F.cross_entropy(logits, target.view(-1))

        return logits, loss

    def generate(self, context: torch.Tensor, num_tokens: int) -> torch.Tensor:
        # generate tokens
        with torch.no_grad():
            for _ in range(num_tokens):
                cond_context = context[:, -self.context_size :]
                logits, _ = self.forward(cond_context)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                context = torch.cat((context, next_token), dim=1)
            return context


class MultiHeadAttention(nn.Module):
    """
    A multi-head attention layer.
    Takes in a number of heads returns a concatenated output of all heads.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        embed_size: int,
        context_size: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, embed_size, context_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x, attention_mask) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class Head(nn.Module):
    """
    A single head of a multi-head attention layer.
    """

    def __init__(
        self, head_size: int, embed_size: int, context_size: int, dropout: float = 0.2
    ):
        super().__init__()
        self.head_size = head_size
        self.queries = nn.Linear(embed_size, head_size, bias=False)
        self.keys = nn.Linear(embed_size, head_size, bias=False)
        self.values = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer(
            "tril_mask", torch.tril(torch.ones(context_size, context_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the head. Takes in a batch of sequences,
        computes the queries, keys, and values, and then computes the attention weights.
        Returns the weighted sum of the values.
        """
        B, T, C = x.shape
        q = self.queries(x)  # (B, T, H)
        k = self.keys(x)  # (B, T, H)
        v = self.values(x)  # (B, T, H)
        weights = (
            q @ k.transpose(-2, -1) / (self.head_size**0.5)
        )  # (B, T, H) @ (B, H, T) -> (B, T, T)
        weights = weights.masked_fill(
            self.tril_mask[:T, :T] == 0, float("-inf")
        )  # (B, T, T)

        # apply attention mask if provided from dataset (for padded tokens)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)  # (B, 1, T)
            attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float).min
            weights = weights + attention_mask

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        return weights @ v  # (B, T, T) @ (B, T, H) -> (B, T, H)


class FeedForward(nn.Module):
    """
    A feed-forward network used in the Transformer.
    """

    def __init__(self, embed_size: int, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        embed_size: int,
        context_size: int,
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            num_heads, head_size, embed_size, context_size
        )
        self.feed_forward = FeedForward(embed_size)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, tuple):
            x, attention_mask = x
        else:
            attention_mask = None
        x = x + self.multi_head_attention(self.layer_norm1(x), attention_mask)
        x = x + self.feed_forward(self.layer_norm2(x))
        return x if attention_mask is None else (x, attention_mask)


if __name__ == "__main__":
    # test the model
    model = DecoderTransformer(2, 2, 32, 8, 100)
    x = torch.randint(0, 100, (32, 8))
    attention_mask = torch.ones(32, 8)
    attention_mask[:, 4:] = 0
    logits, loss = model({"input_ids": x, "attention_mask": attention_mask}, x)
