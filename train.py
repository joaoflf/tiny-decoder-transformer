import os
from datetime import datetime

import time

import fire
import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

from dataset import TinyStoriesDataset
from decoder_transformer import DecoderTransformer


def train(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "mps",
    checkpoint_dir: str = "checkpoints",
    eval_every: int = 100,
):
    device = torch.device(device)
    batch_size = batch_size
    learning_rate = lr
    epochs = epochs
    eval_every = eval_every or epochs // 10

    hidden_size = 512
    context_size = 256
    num_heads = 16
    num_blocks = 8

    train_data = TinyStoriesDataset(
        split="train", context_size=context_size, device=device
    )
    val_data = TinyStoriesDataset(
        split="validation",
        context_size=context_size,
        device=device,
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    eval_iters = len(val_loader)
    start_time = time.time()

    vocab_size = train_data.vocab_size
    model = DecoderTransformer(
        num_blocks=num_blocks,
        num_heads=num_heads,
        hidden_size=hidden_size,
        context_size=context_size,
        vocab_size=vocab_size,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 10e5
    print(f"Loaded model with {total_params:.2f}M parameters")
    model = torch.compile(model) if torch.cuda.is_available() else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb.init(
        project="tiny-decoder-transformer",
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "total_params": total_params,
        },
    )

    # train the model
    for epoch in range(1, epochs + 1):
        with tqdm(train_loader, unit="batch") as progress_bar:
            for i, (x, y) in enumerate(progress_bar):
                progress_bar.set_description(f"Epoch {epoch}")
                model.train()
                optimizer.zero_grad()

                with autocast(enabled=torch.cuda.is_available()):
                    logits, loss = model(x, y)

                    if torch.cuda.is_available():
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                if i % eval_every == 0 and i > 0:
                    # Save the model state
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            checkpoint_dir,
                            f"{time.strftime('%Y%m%d_%H%M', time.localtime(start_time))}_model_state.pt",
                        ),
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
                                    _, loss = model(x, y)
                                    losses[j] = loss.item()
                            eval_losses.append(losses.mean().item())
                    wandb.log(
                        {
                            "Training Loss": eval_losses[0],
                            "Validation Loss": eval_losses[1],
                        }
                    )
                    progress_bar.set_postfix_str(
                        f"Train Loss: {eval_losses[0]:.4f}, Val Loss: {eval_losses[1]:.4f}"
                    )
    training_time = time.time() - start_time
    print(f"Training took {training_time:.2f} seconds")
    wandb.log({"Training Time": training_time})


# run
if __name__ == "__main__":
    fire.Fire(train)
