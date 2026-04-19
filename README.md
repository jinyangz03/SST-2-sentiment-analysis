# SST-2 Sentiment Analysis (Updated)

Modernized from [YJiangcm/SST-2-sentiment-analysis](https://github.com/YJiangcm/SST-2-sentiment-analysis).

## Changes from Original

| Item | Original | Updated |
|------|----------|---------|
| PyTorch | 0.x | 2.x |
| transformers | early 3.x | 4.30+ |
| `forward()` return | tuple unpacking `[:2]` | named attributes `.loss`, `.logits` |
| `AdamW` import | `transformers.optimization` (deprecated) | `torch.optim` |
| Tokenizer | manual `tokenize()` + `convert_tokens_to_ids()` | `tokenizer(...)` batch call |
| Entry point | 5 separate `run_*.py` files | single `run_models.py --model <name>` |
| **New model** | — | **DistilBERT** (`distilbert-base-uncased`) |
| Metrics output | print only | JSON + CSV saved per model |
| ALBERT size | `albert-xxlarge-v2` | `albert-base-v2` (practical for local GPU) |
| XLNet size | `xlnet-large-cased` | `xlnet-base-cased` (practical for local GPU) |

## Models

| Model | Pretrained checkpoint |
|-------|-----------------------|
| BERT | `textattack/bert-base-uncased-SST-2` |
| RoBERTa | `roberta-base` |
| ALBERT | `albert-base-v2` |
| XLNet | `xlnet-base-cased` |
| **DistilBERT** *(new)* | `distilbert-base-uncased` |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train a single model
python run_models.py --model bert
python run_models.py --model distilbert
python run_models.py --model roberta

# Train all models sequentially
python run_models.py --model all

# Custom hyperparameters
python run_models.py --model bert --epochs 5 --batch_size 16 --lr 3e-5
```

## Output Structure

```
output/
  bert/
    best.pth.tar        # best checkpoint
    metrics.json        # per-epoch train/val metrics + test results
    test_prediction.csv # test set predictions
  roberta/
    ...
  summary.json          # cross-model comparison table
```

## Data

Place SST-2 `.tsv` files in `./data/`:
- `train.tsv` — training set
- `dev.tsv` — validation set  
- `test.tsv` — test set (with labels, column 0 = label, column 1 = sentence)

## Notes

- DistilBERT and RoBERTa do not use `token_type_ids`; these are zeroed-out automatically.
- Linear warmup scheduler (10% of steps by default) replaces the original `ReduceLROnPlateau`.
- All models save `metrics.json` which task B visualization scripts consume directly.
