from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast as Tokenizer


class TinyStoriesDataset(Dataset):
    def __init__(
        self, split: str = "train", sequence_length: int = 256, device: str = "mps"
    ):
        self.dataset = load_dataset("roneneldan/TinyStories", split=split)
        self.tokenizer = Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sequence_length = sequence_length
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]["input_ids"][:-1]
        y = self.dataset[idx]["input_ids"][1:]
        return x, y


if __name__ == "__main__":
    val_dataset = TinyStoriesDataset(split="validation")
    loader = DataLoader(val_dataset, batch_size=1)
    pass
