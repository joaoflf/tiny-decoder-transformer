from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast as Tokenizer


class TinyStoriesDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        context_size: int = 256,
        device: torch.device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps"),
    ):
        self.device = device
        self.dataset = load_dataset("roneneldan/TinyStories", split=split)
        self.tokenizer = Tokenizer.from_pretrained("gpt2", local_files_only=True)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.sequence_length = context_size
        self.dataset = self.dataset.with_format(type="torch")
        self.dataset = self.dataset.map(
            lambda x: self.tokenizer(
                x["text"],
                truncation=True,
                padding="max_length",
                max_length=self.sequence_length,
            ),
            batched=True,
        )
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[dict, torch.Tensor]:
        x = {
            "input_ids": self.dataset[idx]["input_ids"][:-1].to(self.device),
            "attention_mask": self.dataset[idx]["attention_mask"][:-1].to(self.device),
        }
        y = self.dataset[idx]["input_ids"][1:].to(self.device)
        return x, y
