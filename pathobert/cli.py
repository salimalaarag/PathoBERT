#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3

"""
Command-line interface for PathoBERT.
"""
import os
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATALOADER_WORKERS,
    INPUT_IDS_FILE,
    ATTENTION_MASK_FILE,
)

from .fasta import read_fasta
from .tokenizer import tokenize_sequences
from .dataset import IterableChunkDataset
from .model import load_model
from .inference import predict
from .analysis import analyze_predictions

#save_dir = "./tokenized"
#os.makedirs(save_dir, exist_ok=True)

OUTPUT_DIR = "./output"
TOKEN_DIR = os.path.join(OUTPUT_DIR, "tokenized")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TOKEN_DIR, exist_ok=True)

def build_parser():
    parser = argparse.ArgumentParser(
        prog="pathobert",
        description="PathoBERT read-level bacterial pathogenicity prediction",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained checkpoint (.pt)",
    )

    parser.add_argument(
        "--fasta",
        required=True,
        help="Input FASTA file",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_DATALOADER_WORKERS,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--out",
        default="predictions.tsv",
        help="Output TSV file",
    )

    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Do not display histogram",
    )

    return parser


def main():

    args = build_parser().parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("📂 Reading FASTA...")
    ids, sequences = read_fasta(args.fasta)

    print("🧬 Tokenizing...")
    results = tokenize_sequences(sequences)
    input_ids_list, attention_mask_list = results

    input_ids_array = np.stack(input_ids_list)
    attention_mask_array = np.stack(attention_mask_list)
    total_size = input_ids_array.shape[0]
    print(
        f"💾 Saved tensors: "
        f"{input_ids_array.shape}"
    )
    # Save
    np.save(os.path.join(TOKEN_DIR, "input_ids.npy"), input_ids_array)
    np.save(os.path.join(TOKEN_DIR, "attention_mask.npy"), attention_mask_array)
    del input_ids_array, attention_mask_array

    

    dataset = IterableChunkDataset(
        input_ids_path=INPUT_IDS_FILE,
        attention_mask_path=ATTENTION_MASK_FILE,
        shuffle=False,
        rank=0,
        world_size=1,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("📥 Loading model...")
    model = load_model(args.model, device)

    print("🚀 Running inference...")
    probs = predict(
        model=model,
        loader=loader,
        total_reads=len(ids),
        batch_size=args.batch_size,
        device=device,
    )
    if isinstance(probs, list):
       probs = torch.cat(probs).numpy()
    elif isinstance(probs, torch.Tensor):
       probs = probs.cpu().numpy()

    torch.save(torch.tensor(probs), os.path.join(OUTPUT_DIR, "probs.pt"))
    print("💾 Saving probs...")

    print("📊 Analyzing predictions...")
    stats = analyze_predictions(
        probs,
        threshold=args.threshold,
        show_plot=not args.no_plot,
    )
    torch.save(stats, os.path.join(OUTPUT_DIR, "stats.pt"))
    print("💾 Saving results...")
    output_file = os.path.join(OUTPUT_DIR,args.out)

    with open(output_file, "w") as f:
        f.write("sequence\tprobability\tlabel\n")

        for seq, prob in zip(sequences, probs):
            label = "PATHOGEN" if prob > 0.5 else "NON-PATHOGEN"
            f.write(f"{seq}\t{prob:.6f}\t{label}\n")

    print(f"✅ Results saved in: {OUTPUT_DIR}")
    print(f"📄 Prediction file: {output_file}")

    print("✅ Finished.")


if __name__ == "__main__":
    main()

