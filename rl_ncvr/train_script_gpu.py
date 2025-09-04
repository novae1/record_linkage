"""
Single-GPU training script compatible with the dataset format expected by
rl_ncvr/train_script.py, but without TPU/XLA. It mirrors the CLI and behavior
where practical, trains with symmetric InfoNCE for 2-col pairs.

Usage (run from project root):
    python rl_ncvr/train_script_gpu.py \
        --model nreimers/MiniLM-L6-H384-uncased \
        --steps 2000 \
        --save_steps 10000 \
        --batch_size 64 \
        --max_length 128 \
        --datasets_per_batch 2 \
        --scale 20 \
        --data_folder rl_ncvr/outputs \
        rl_ncvr/outputs/data_config.json \
        outputs/run_minilm_box0_gpu

data_config.json example:
    [
      {"name": "box0_positive_pairs.json.gz", "weight": 1}
    ]
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import random
from typing import Dict, Iterable, List

import torch
from torch import nn

from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)


class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self, model_name: str, tokenizer, normalize: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.tokenizer = tokenizer

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def save_pretrained(self, output_path: str) -> None:
        self.tokenizer.save_pretrained(output_path)
        self.model.config.save_pretrained(output_path)
        torch.save(self.model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))


class Dataset:
    """Stream one .json.gz dataset, optionally caching if small."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def __iter__(self):
        max_dataset_size = 10 * 1000 * 1000
        dataset = []
        data_format = None

        while dataset is None or len(dataset) == 0:
            with gzip.open(self.filepath, "rt") as f_in:
                for line in f_in:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        data = data['texts']

                    if data_format is None:
                        data_format = len(data)

                    assert len(data) == data_format

                    if dataset is not None:
                        dataset.append(data)
                        if len(dataset) >= max_dataset_size:
                            dataset = None

                    yield data

        while True:
            random.shuffle(dataset)
            for data in dataset:
                yield data


def load_datasets(data_folder: str, data_config_path: str):
    with open(data_config_path) as f_in:
        data_config = json.load(f_in)
    filepaths = []
    dataset_indices = []
    for idx, data in enumerate(data_config):
        filepaths.append(os.path.join(os.path.expanduser(data_folder), data['name']))
        dataset_indices.extend([idx] * data['weight'])

    datasets = [iter(Dataset(fp)) for fp in filepaths]
    # Determine column count per dataset (2 vs 3)
    num_cols = {}
    for i, it in enumerate(datasets):
        sample = next(it)
        num_cols[i] = len(sample)
        # re-chain the sample back by wrapping an iterator
        def prepend_sample(first, it_rest):
            yield first
            for x in it_rest:
                yield x
        datasets[i] = prepend_sample(sample, it)
    return datasets, dataset_indices, num_cols


def build_batch(args, datasets, dataset_indices, num_cols):
    texts_in_batch = set()
    batch = []
    batch_format = None

    size_per_dataset = args.batch_size  # single GPU -> one dataset slot at a time

    while len(batch) < args.batch_size:
        # Choose a dataset index respecting format consistency
        valid_dataset = False
        while not valid_dataset:
            data_idx = random.choice(dataset_indices)
            if batch_format is None:
                batch_format = num_cols[data_idx]
                valid_dataset = True
            else:
                valid_dataset = (batch_format == num_cols[data_idx])

        dataset = datasets[data_idx]
        # Pull one sample that doesn't duplicate any text in the batch
        while True:
            sample = next(dataset)
            in_batch = False
            for text in sample:
                if text in texts_in_batch:
                    in_batch = True
                    break
            if not in_batch:
                for text in sample:
                    texts_in_batch.add(text)
                batch.append(sample)
                break

    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='nreimers/MiniLM-L6-H384-uncased')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--save_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--datasets_per_batch', type=int, default=2, help="Ignored on single GPU; kept for CLI parity")
    parser.add_argument('--scale', type=float, default=20, help="Use 20 for cossim, and 1 for unnormalized embeddings with dot product")
    parser.add_argument('--data_folder', default='rl_ncvr/outputs', help='Folder with your dataset files')
    parser.add_argument('data_config', help='A data_config.json file')
    parser.add_argument('output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('CUDA GPU not available. Please ensure an NVIDIA GPU is accessible.')

    os.makedirs(args.output, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSentenceEmbedding(args.model, tokenizer).to(device)

    optimizer = AdamW(params=model.parameters(), lr=2e-5, correct_bias=True)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.steps,
    )

    cross_entropy_loss = nn.CrossEntropyLoss()
    max_grad_norm = 1.0

    datasets, dataset_indices, num_cols = load_datasets(args.data_folder, args.data_config)

    model.train()
    for global_step in range(args.steps):
        batch = build_batch(args, datasets, dataset_indices, num_cols)

        if len(batch[0]) == 2:
            text1 = tokenizer([b[0] for b in batch], return_tensors='pt', max_length=args.max_length, truncation=True, padding='max_length')
            text2 = tokenizer([b[1] for b in batch], return_tensors='pt', max_length=args.max_length, truncation=True, padding='max_length')

            embeddings_a = model(**{k: v.to(device) for k, v in text1.items()})
            embeddings_b = model(**{k: v.to(device) for k, v in text2.items()})

            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
            labels = torch.arange(len(scores), dtype=torch.long, device=device)
            loss = (cross_entropy_loss(scores, labels) + cross_entropy_loss(scores.t(), labels)) / 2

        else:  # len == 3
            text1 = tokenizer([b[0] for b in batch], return_tensors='pt', max_length=args.max_length, truncation=True, padding='max_length')
            text2 = tokenizer([b[1] for b in batch], return_tensors='pt', max_length=args.max_length, truncation=True, padding='max_length')
            text3 = tokenizer([b[2] for b in batch], return_tensors='pt', max_length=args.max_length, truncation=True, padding='max_length')

            embeddings_a  = model(**{k: v.to(device) for k, v in text1.items()})
            embeddings_b1 = model(**{k: v.to(device) for k, v in text2.items()})
            embeddings_b2 = model(**{k: v.to(device) for k, v in text3.items()})

            embeddings_b = torch.cat([embeddings_b1, embeddings_b2])
            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
            labels = torch.arange(len(scores), dtype=torch.long, device=device)
            loss = cross_entropy_loss(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()

        if (global_step + 1) % args.save_steps == 0:
            output_path = os.path.join(args.output, str(global_step + 1))
            os.makedirs(output_path, exist_ok=True)
            print("save model:", output_path)
            model.save_pretrained(output_path)

    final_path = os.path.join(args.output, 'final')
    os.makedirs(final_path, exist_ok=True)
    print('save model final:', final_path)
    model.save_pretrained(final_path)


if __name__ == '__main__':
    main()


