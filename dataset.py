import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast as Tokenizer


class TinyStoriesDataset(Dataset):
    def __init__(self, split: str = "train", context_size: int = 256):
        self.dataset = load_dataset("roneneldan/TinyStories", split=split)
        self.tokenizer = Tokenizer.from_pretrained("gpt2", local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sequence_length = context_size
        self.dataset = self.dataset.with_format(type="torch")
        self.dataset = self.dataset.map(
            lambda x: self.tokenizer(
                x["text"],
                truncation=True,
                padding="max_length",
                max_length=self.sequence_length + 1,
            ),
            batched=True,
        )
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]["input_ids"][:-1]
        y = self.dataset[idx]["input_ids"][1:]
        return x, y
