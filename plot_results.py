import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

OUTPUT_DIR = "output"
FIG_DIR = "figures"

os.makedirs(FIG_DIR, exist_ok=True)

model_names = ["bert", "roberta", "albert", "xlnet", "distilbert"]
summary_rows = []
f1_rows = []

for model in model_names:
    metrics_path = os.path.join(OUTPUT_DIR, model, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"Skip {model}: metrics.json not found")
        continue

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    epochs = data["epochs"]
    train_losses = data["train_losses"]
    train_accs = data["train_accuracies"]
    val_losses = data["valid_losses"]
    val_accs = data["valid_accuracies"]
    y_true = data["test_labels"]
    y_pred = data["test_preds"]

    # 1) Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, marker="o", label="Train Accuracy")
    plt.plot(epochs, val_accs, marker="o", label="Validation Accuracy")
    plt.title(f"{model.upper()} Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{model}_accuracy_curve.png"))
    plt.close()

    # 2) Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation Loss")
    plt.title(f"{model.upper()} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{model}_loss_curve.png"))
    plt.close()

    # 3) Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(f"{model.upper()} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{model}_confusion_matrix.png"))
    plt.close()

    # 4) F1 per class
    f1_neg = f1_score(y_true, y_pred, pos_label=0)
    f1_pos = f1_score(y_true, y_pred, pos_label=1)

    f1_rows.append({
        "model": model,
        "negative": f1_neg,
        "positive": f1_pos,
    })

    summary_rows.append({
    "dataset": "SST-2",
    "model": model,
    "test_acc": accuracy_score(y_true, y_pred),
    "val_acc": data["best_val_acc"],
    "val_loss": min(data["valid_losses"]),
    "train_time": data["total_train_time_sec"],
    "epochs": len(data["epochs"]),
    "num_classes": 2,
    "f1_negative": f1_neg,
    "f1_positive": f1_pos,
    "split": "balanced"
})

# 5) Summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

# 6) Test accuracy bar chart
if not summary_df.empty:
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["model"], summary_df["test_acc"])
    plt.title("Test Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "test_accuracy_bar.png"))
    plt.close()

    # 7) Accuracy vs training time
    plt.figure(figsize=(8, 5))
    plt.scatter(summary_df["train_time"], summary_df["test_acc"])
    for _, row in summary_df.iterrows():
        plt.text(row["train_time"], row["test_acc"], row["model"])
    plt.title("Accuracy vs Training Time")
    plt.xlabel("Training Time (sec)")
    plt.ylabel("Test Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "accuracy_vs_training_time.png"))
    plt.close()

# 8) F1 heatmap
f1_df = pd.DataFrame(f1_rows)
if not f1_df.empty:
    f1_df = f1_df.set_index("model")

    plt.figure(figsize=(6, 4))
    plt.imshow(f1_df.values)
    plt.title("F1 Score Heatmap")
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks(range(len(f1_df.index)), f1_df.index)

    for i in range(f1_df.shape[0]):
        for j in range(f1_df.shape[1]):
            plt.text(j, i, f"{f1_df.iloc[i, j]:.4f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "f1_score_heatmap.png"))
    plt.close()

print("Done!")
print(f"Figures saved in: {FIG_DIR}")
print(f"Summary CSV saved in: {summary_csv_path}")