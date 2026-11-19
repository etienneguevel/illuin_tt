import json

import numpy as np
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    (text, label) pairs read from the JSON files. With a list of fields, `texts` is a
    {field: [texts]} dict and items are ({field: text}, label).
    """

    def __init__(self, files, labels, field: str | list[str]):
        super().__init__()
        self.files = [str(f) for f in np.asarray(files).ravel()]
        self.labels = np.asarray(labels, dtype=np.float32)
        if len(self.labels) != len(self.files):
            raise ValueError(f"Got {len(self.files)} files but {len(self.labels)} labels")

        self.num_classes = self.labels.shape[1]
        self.field = field
        fields = [field] if isinstance(field, str) else list(field)

        texts = {f: [] for f in fields}
        for file in self.files:
            with open(file) as fp:
                d = json.load(fp)
            for f in fields:
                texts[f].append(d[f])

        self.texts = texts[field] if isinstance(field, str) else texts

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        if isinstance(self.texts, dict):
            return {f: t[index] for f, t in self.texts.items()}, self.labels[index]
        return self.texts[index], self.labels[index]
