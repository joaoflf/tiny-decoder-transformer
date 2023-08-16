import fire
import torch
from tqdm import tqdm
from dataset import TinyStoriesDataset
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast as Tokenizer
from decoder_transformer import DecoderTransformer
from torch.cuda.amp.autocast_mode import autocast
import yaml


def evaluate(
    checkpoint_path: str,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "mps",
    num_tokens: int = 100,
    prompt: str = """Tom and Jane are friends. One day, Jane goes to Tom’s house. Tom has a big pot of soup. He wants to share it with Jane. “Jane, do you want some soup?” Tom asks. “Yes, please. It looks yummy,” Jane says. Tom pours some soup into two bowls. He gives one bowl to Jane. Jane takes a spoonful of soup, but then she makes a face. The soup is""",
    batch_size: int = 32,
    check_train_loss: bool = False,
    check_val_loss: bool = False,
    model_version: str = "77M",
    multi_gpus: bool = False,
):
    tokenizer = Tokenizer.from_pretrained("gpt2", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(device)
    vocab_size = tokenizer.vocab_size

    with open("model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)

    model_config = model_config[model_version]
    hidden_size = model_config["hidden_size"]
    context_size = model_config["context_size"]
    num_heads = model_config["num_heads"]
    num_blocks = model_config["num_blocks"]

    # Load the model from the checkpoint
    model = DecoderTransformer(
        num_blocks=num_blocks,
        num_heads=num_heads,
        hidden_size=hidden_size,
        context_size=context_size,
        vocab_size=vocab_size,
    ).to(device)
    if multi_gpus:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    # model = torch.compile(model)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.eval()
    splits = {}

    if check_train_loss:
        train_data = TinyStoriesDataset(
            split="train", context_size=context_size, device=device
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        splits.update({"training": train_loader})
    if check_val_loss:
        val_data = TinyStoriesDataset(
            split="validation", context_size=context_size, device=device
        )
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        splits.update({"validation": val_loader})

    for split, loader in splits.items():
        print(f"Evaluating on {split} set")
        total_loss = 0
        for x, y in tqdm(loader):
            with autocast(enabled=torch.cuda.is_available()):
                x, y = x, y
                with torch.no_grad():
                    _, loss = model(x, y)
                    if multi_gpus:
                        loss = loss.mean()
                total_loss += loss.item()
        total_loss /= len(loader)
        print(f"\n{split} loss: {total_loss:.2f}")

    print("\n\n" + "=" * 80)
    encoded_context = (
        torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    )

    print(prompt, end="")
    # Generate text
    if multi_gpus:
        model = model.module
    generated_text = model.generate(encoded_context, num_tokens)
    for token in generated_text:
        print(tokenizer.decode(token.tolist()[0]), end="")


if __name__ == "__main__":
    fire.Fire(evaluate)
