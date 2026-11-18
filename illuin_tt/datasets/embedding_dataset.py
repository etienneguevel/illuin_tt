import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from tqdm import tqdm

from illuin_tt.utils import get_device


def text_yielder(files: list[str], tokenizer, max_tokens: int, key: str, yield_last: bool = True):
    """
    Yield (texts, indices) batches of at most max_tokens tokens in total.
    `indices` are the positions in `files` of the texts in the batch. Files whose
    text alone exceeds max_tokens are skipped and never appear in any batch.
    """
    texts, indices = [], []
    char_count = 0

    for i, file in enumerate(files):
        with open(file) as f:
            d = json.load(f)

        txt = d[key]
        n_tok = len(tokenizer(txt)["input_ids"])
        if n_tok > max_tokens:
            print(f"Skipping file {file} for len of {n_tok}")
            continue

        if char_count + n_tok > max_tokens and texts:
            yield texts, indices
            texts, indices, char_count = [], [], 0

        texts.append(txt)
        indices.append(i)
        char_count += n_tok

    if yield_last and texts:
        yield texts, indices


class EmbeddingDataset(Dataset):

    def __init__(
        self,
        model_name: str,
        files: list[str],
        labels,
        field: str,
        cache_path: str,
        num_classes: int,
        max_tokens: int = 25000,
        device=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.field = field
        self.device = get_device(device)

        # Accept plain lists as well as the (n, 1) path arrays from make_train_test_val
        all_files = [str(f) for f in np.asarray(files).ravel()]
        all_labels = np.asarray(labels, dtype=np.float32)
        if len(all_labels) != len(all_files):
            raise ValueError(f"Got {len(all_files)} files but {len(all_labels)} labels")

        self.files = all_files

        # Load the cache and meta_data if it exists
        meta_path = os.path.splitext(cache_path)[0] + ".json"
        loaded = self._load_cache(cache_path)
        if loaded is None:
            self.cache, kept = self._compute_embeddings(max_tokens)
        else:
            self.cache, kept = loaded

        # Drop the files (and their labels) whose text exceeded max_tokens
        kept_set = set(kept)
        self.skipped = [f for i, f in enumerate(all_files) if i not in kept_set]
        self.files = [all_files[i] for i in kept]
        self.labels = all_labels[kept]
        if self.skipped:
            print(f"Skipped {len(self.skipped)} files longer than max_tokens")

        if loaded is None:
            print(f"Successfully computed embeddings, saving at: {cache_path}")
            self._save_cache(cache_path, meta_path)

        self.cache = self.cache.astype(np.float32)
        self.emb_dim = self.cache.shape[1]

    def _load_cache(self, cache_path: str) -> tuple[np.ndarray, list[int]] | None:
        """
        Assemble embeddings for self.files from every cache in the same directory
        built with the same model (per-file lookup, so a re-split reuses old caches).
        Files recorded as skipped (too long) in a sidecar are dropped; returns the
        embeddings and the indices in self.files of the kept files.
        Note: skipped files are not tied to max_tokens; clear the cache if it changes.
        """
        cache_dir = os.path.dirname(cache_path)
        if not os.path.isdir(cache_dir):
            return None

        sources = {}
        skipped = set()
        for name in sorted(os.listdir(cache_dir)):
            if not name.endswith(".json"):
                continue

            # Check if there is the npy cache file
            npy = os.path.join(cache_dir, name[:-5] + ".npy")
            if not os.path.exists(npy):
                continue

            # Check that the meta information are the same as current init cfg
            with open(os.path.join(cache_dir, name)) as f:
                meta = json.load(f)
            if meta.get("model_name") != self.model_name:
                continue

            # Recover skipped files
            for row, file in enumerate(meta.get("files", [])):
                sources.setdefault(file, (npy, row))
            skipped.update(meta.get("skipped", []))

        # Check no files are left out
        missing = [f for f in self.files if f not in sources and f not in skipped]
        if missing:
            print(f"{len(missing)} files not found in cache for {self.model_name}, recomputing.")
            return None

        # Recover the files in the different saved caches, useful if the seed has changed
        arrays = {}
        rows, kept = [], []
        for i, file in enumerate(self.files):
            if file not in sources:
                continue
            npy, row = sources[file]
            if npy not in arrays:
                arrays[npy] = np.load(npy, mmap_mode="r")
            rows.append(arrays[npy][row])
            kept.append(i)
        if not rows:
            return None
        cache = np.stack(rows)

        print(f"Loaded cached embeddings for {cache_path}")
        return cache, kept


    def _save_cache(self, cache_path: str, meta_path: str):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, self.cache)

        with open(meta_path, "w") as f:
            json.dump(
                {"model_name": self.model_name, "files": self.files, "skipped": self.skipped}, f
            )


    def _compute_embeddings(self, max_tokens: int) -> tuple[np.ndarray, list[int]]:
        embs, kept = [], []
        model = SentenceTransformer(self.model_name, device=str(self.device))

        batch_gen = text_yielder(self.files, model.tokenizer, max_tokens, self.field)

        for texts, indices in tqdm(batch_gen):
            embs.append(model.encode(texts))
            kept.extend(indices)

        embs = np.concatenate(embs)
        return embs, kept

    
    def __len__(self):
        return len(self.files)


    def __getitem__(self, index: int):
        # float32 arrays so default_collate yields (bs, emb_dim) / (bs, num_classes) tensors
        return self.cache[index], self.labels[index]

