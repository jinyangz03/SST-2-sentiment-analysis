# -*- coding: utf-8 -*-
"""
run_models.py
=============
Unified training + evaluation script for SST-2 sentiment analysis.

Models supported:
  bert, roberta, albert, xlnet, distilbert

Usage:
  python run_models.py --model bert
  python run_models.py --model distilbert --epochs 3 --batch_size 32
  python run_models.py --model all   # train all models sequentially

Updated from original YJiangcm codebase:
  - Single entry point instead of separate run_*.py files
  - PyTorch 2.x + transformers 4.x compatible
  - AdamW from torch.optim (not transformers.optimization, which is deprecated)
  - Returns structured results for downstream visualization (task B)
  - Saves per-epoch metrics to JSON for reproducibility
"""

import os
import json
import argparse
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from data_sst2 import DataPrecessForSentence
from utils import train, validate, test, Metric
from models import BertModel, RobertModel, AlbertModel, XlnetModel, DistilBertModel

MODEL_MAP = {
    "bert": BertModel,
    "roberta": RobertModel,
    "albert": AlbertModel,
    "xlnet": XlnetModel,
    "distilbert": DistilBertModel,
}


def model_train_validate_test(
    model_name,
    train_df,
    dev_df,
    test_df,
    target_dir,
    max_seq_len=50,
    epochs=3,
    batch_size=32,
    lr=2e-5,
    patience=2,
    max_grad_norm=10.0,
    if_save_model=True,
    checkpoint=None,
    warmup_ratio=0.1,
):
    """
    Train, validate, and test a model on SST-2.
    Returns a results dict with all metrics for downstream plotting.
    """
    assert model_name in MODEL_MAP, f"Unknown model: {model_name}. Choose from {list(MODEL_MAP)}"

    os.makedirs(target_dir, exist_ok=True)

    # ---------- Build model & tokenizer ----------
    print(f"\n{'='*20} Building {model_name.upper()} {'='*20}")
    model_obj = MODEL_MAP[model_name](requires_grad=True)
    tokenizer = model_obj.tokenizer
    device = model_obj.device
    model = model_obj.to(device)

    # ---------- Data loaders ----------
    print("Loading data...")
    train_data = DataPrecessForSentence(tokenizer, train_df, max_seq_len=max_seq_len)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=0)

    dev_data = DataPrecessForSentence(tokenizer, dev_df, max_seq_len=max_seq_len)
    dev_loader = DataLoader(dev_data, shuffle=False, batch_size=batch_size, num_workers=0)

    test_data = DataPrecessForSentence(tokenizer, test_df, max_seq_len=max_seq_len)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=0)

    # ---------- Optimizer ----------
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    total_steps = len(train_loader) * epochs
    num_warmup = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup, num_training_steps=total_steps
    )

    # ---------- Tracking ----------
    best_score = 0.0
    patience_counter = 0
    start_epoch = 1

    epochs_count = []
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies, valid_aucs = [], [], []

    total_train_time = 0.0  # cumulative training time across epochs

    # Resume from checkpoint
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt["best_score"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        epochs_count = ckpt.get("epochs_count", [])
        train_losses = ckpt.get("train_losses", [])
        train_accuracies = ckpt.get("train_accuracy", [])
        valid_losses = ckpt.get("valid_losses", [])
        valid_accuracies = ckpt.get("valid_accuracy", [])
        valid_aucs = ckpt.get("valid_auc", [])
        total_train_time = ckpt.get("total_train_time", 0.0)
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    # Pre-training validation
    _, valid_loss, valid_acc, auc, _ = validate(model, dev_loader)
    print(f"Pre-training: val_loss={valid_loss:.4f}, val_acc={valid_acc:.4f}, auc={auc:.4f}")

    # ---------- Training loop ----------
    print(f"\n{'='*20} Training {model_name.upper()} on {device} {'='*20}")
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        print(f"\n--- Epoch {epoch}/{epochs} ---")
        epoch_time, epoch_loss, epoch_acc = train(model, train_loader, optimizer, epoch, max_grad_norm)
        total_train_time += epoch_time
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f"Train: time={epoch_time:.1f}s, loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

        scheduler.step()

        _, val_loss, val_acc, val_auc, _ = validate(model, dev_loader)
        valid_losses.append(val_loss)
        valid_accuracies.append(val_acc)
        valid_aucs.append(val_auc)
        print(f"Valid: loss={val_loss:.4f}, acc={val_acc:.4f}, auc={val_auc:.4f}")

        # Save best model
        if val_acc >= best_score:
            best_score = val_acc
            patience_counter = 0
            if if_save_model:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "train_accuracy": train_accuracies,
                        "valid_losses": valid_losses,
                        "valid_accuracy": valid_accuracies,
                        "valid_auc": valid_aucs,
                        "total_train_time": total_train_time,
                    },
                    os.path.join(target_dir, "best.pth.tar"),
                )
                print(f"Saved best model (val_acc={best_score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # ---------- Test ----------
    print(f"\n{'='*20} Testing {model_name.upper()} {'='*20}")
    batch_time, total_test_time, test_acc, all_prob, all_preds, all_labels = test(model, test_loader)
    print(f"Test accuracy: {test_acc:.4f}")
    Metric(all_labels, all_preds)

    # Save test predictions
    test_prediction = pd.DataFrame(
        {"prob_0": [1 - p for p in all_prob], "prob_1": all_prob, "prediction": all_preds, "label": all_labels}
    )
    test_prediction.to_csv(os.path.join(target_dir, "test_prediction.csv"), index=False)

    # Save epoch metrics to JSON (for task B visualization)
    metrics = {
        "model": model_name,
        "epochs": epochs_count,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "valid_losses": valid_losses,
        "valid_accuracies": valid_accuracies,
        "valid_aucs": valid_aucs,
        "best_val_acc": best_score,
        "test_acc": test_acc,
        "total_train_time_sec": total_train_time,
        "test_preds": [int(p) for p in all_preds],
        "test_labels": [int(l) for l in all_labels],
    }
    with open(os.path.join(target_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Results saved to {target_dir}")
    return metrics


def load_data(data_dir):
    train_df = pd.read_csv(
        os.path.join(data_dir, "train.tsv"), sep="\t", header=None, names=["similarity", "s1"]
    )
    dev_df = pd.read_csv(
        os.path.join(data_dir, "dev.tsv"), sep="\t", header=None, names=["similarity", "s1"]
    )
    test_df = pd.read_csv(
        os.path.join(data_dir, "test.tsv"), sep="\t", header=None, names=["similarity", "s1"]
    )
    return train_df, dev_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SST-2 Sentiment Analysis Trainer")
    parser.add_argument("--model", type=str, default="bert",
                        help="Model to train: bert | roberta | albert | xlnet | distilbert | all")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    train_df, dev_df, test_df = load_data(args.data_dir)

    models_to_run = list(MODEL_MAP.keys()) if args.model == "all" else [args.model]

    all_results = {}
    for model_name in models_to_run:
        target_dir = os.path.join(args.output_dir, model_name)
        results = model_train_validate_test(
            model_name=model_name,
            train_df=train_df,
            dev_df=dev_df,
            test_df=test_df,
            target_dir=target_dir,
            max_seq_len=args.max_seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            max_grad_norm=args.max_grad_norm,
            if_save_model=not args.no_save,
            warmup_ratio=args.warmup_ratio,
        )
        all_results[model_name] = results

    # Save summary
    summary = {
        name: {
            "test_acc": r["test_acc"],
            "best_val_acc": r["best_val_acc"],
            "total_train_time_sec": r["total_train_time_sec"],
        }
        for name, r in all_results.items()
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\n====== FINAL SUMMARY ======")
    for name, s in summary.items():
        print(f"{name:12s}: test_acc={s['test_acc']:.4f}, val_acc={s['best_val_acc']:.4f}, time={s['total_train_time_sec']:.1f}s")
