import fire
import torch
from tqdm import tqdm
from dataset import TinyStoriesDataset
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast as Tokenizer
from decoder_transformer import DecoderTransformer
from torch.cuda.amp.autocast_mode import autocast


def evaluate(
    checkpoint_path: str,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "mps",
    num_tokens: int = 100,
    prompt: str = "Sol had a dog named Jobim",
    batch_size: int = 32,
    check_train_loss: bool = False,
    check_val_loss: bool = False,
):
    tokenizer = Tokenizer.from_pretrained("gpt2", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(device)
    vocab_size = tokenizer.vocab_size
    hidden_size = 256
    context_size = 256
    num_heads = 2
    num_blocks = 2

    # Load the model from the checkpoint
    model = DecoderTransformer(
        num_blocks=num_blocks,
        num_heads=num_heads,
        hidden_size=hidden_size,
        context_size=context_size,
        vocab_size=vocab_size,
    ).to(device)
    model = torch.compile(model)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.eval()

    train_data = TinyStoriesDataset(
        split="train", context_size=context_size, device=device
    )
    val_data = TinyStoriesDataset(
        split="validation", context_size=context_size, device=device
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    splits = {}
    if check_train_loss:
        splits.update({"training": train_loader})
    if check_val_loss:
        splits.update({"validation": val_loader})
    for split, loader in splits.items():
        print(f"Evaluating on {split} set")
        total_loss = 0
        for x, y in tqdm(loader):
            with autocast(enabled=torch.cuda.is_available()):
                x, y = x, y
                with torch.no_grad():
                    _, loss = model(x, y)
                total_loss += loss.item()
        total_loss /= len(loader)
        print(f"\n{split} loss: {total_loss:.2f}")

    print("\n\n" + "=" * 80)
    encoded_context = (
        torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    )

    # Generate text
    generated_text = model.generate(encoded_context, num_tokens)
    print(tokenizer.decode(generated_text.tolist()[0]))


if __name__ == "__main__":
    fire.Fire(evaluate)
