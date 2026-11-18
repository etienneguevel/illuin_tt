import json

import numpy as np
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, files, labels, field: str):
        super().__init__()
        self.files = [str(f) for f in np.asarray(files).ravel()]
        self.labels = np.asarray(labels, dtype=np.float32)
        if len(self.labels) != len(self.files):
            raise ValueError(f"Got {len(self.files)} files but {len(self.labels)} labels")

        self.num_classes = self.labels.shape[1]
        self.field = field

        self.texts = []
        for file in self.files:
            with open(file) as f:
                self.texts.append(json.load(f)[field])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index: int):
        return self.texts[index], self.labels[index]
