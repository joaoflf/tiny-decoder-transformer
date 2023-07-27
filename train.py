import argparse
import os
from datetime import datetime

import fire
import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import TinyStoriesDataset
from decoder_transformer import DecoderTransformer


def train(
    iters: int = 1000,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "mps",
    checkpoint_dir: str = "checkpoints",
    eval_every: int = 100,
):
    device = torch.device(device)
    batch_size = batch_size
    learning_rate = lr
    iters = iters
    eval_every = eval_every or iters // 10
    eval_iters = iters // 10

    hidden_size = 768
    context_size = 256
    num_heads = 2
    num_blocks = 2

    train_data = TinyStoriesDataset(split="train", context_size=context_size)
    val_data = TinyStoriesDataset(split="validation", context_size=context_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    start_time = datetime.now().strftime("%Y%m%d_%H%M")

    vocab_size = train_data.vocab_size
    model = DecoderTransformer(
        num_blocks=num_blocks,
        num_heads=num_heads,
        hidden_size=hidden_size,
        context_size=context_size,
        vocab_size=vocab_size,
    ).to(device)
    print(
        f"Loaded model with {sum(p.numel() for p in model.parameters())/10e5:.2f}M parameters"
    )

    model = torch.compile(model) if torch.cuda.is_available() else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    progress_bar = tqdm(range(1, iters + 1))
    os.makedirs(checkpoint_dir, exist_ok=True)

    # train the model
    train_loader_iter = iter(train_loader)
    for i in progress_bar:
        x, y = next(train_loader_iter)
        model.train()
        optimizer.zero_grad()

        with autocast(enabled=torch.cuda.is_available()):
            logits, loss = model(x.to(device), y.to(device))

            if torch.cuda.is_available():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        if i % eval_every == 0:
            # Save the model state
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, f"{start_time}_model_state.pt"),
            )
            # Evaluate the model
            model.eval()
            with torch.no_grad():
                eval_losses = []
                for loader in [iter(train_loader), iter(val_loader)]:
                    losses = torch.zeros(eval_iters)
                    for j in range(eval_iters):
                        with autocast(enabled=torch.cuda.is_available()):
                            x, y = next(loader)
                            model.eval()
                            _, loss = model(x.to(device), y.to(device))
                            losses[j] = loss.item()
                    eval_losses.append(losses.mean().item())
            progress_bar.set_postfix_str(
                f"Train Loss: {eval_losses[0]:.4f}, Val Loss: {eval_losses[1]:.4f}"
            )


# run
if __name__ == "__main__":
    fire.Fire(train)
