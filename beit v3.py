import os
import json
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import BeitForImageClassification
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, average_precision_score,
    hamming_loss, accuracy_score, matthews_corrcoef, cohen_kappa_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from itertools import cycle

try:
    import optuna
except ImportError:
    optuna = None

# === CONFIGURATION ===
def _discover_workspace_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "MetaData_Cleaned.csv").exists() and (candidate / "Indian Medicinal Leaves").exists():
            return candidate
    return current


PROJECT_DIR = _discover_workspace_root()
IMAGE_PATH = PROJECT_DIR / "Indian Medicinal Leaves"
SAVE_DIR = PROJECT_DIR / "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

RUN_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(log_dir=os.path.join(PROJECT_DIR, "runs", f"BEiT_{RUN_STAMP}"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

cpu_workers = os.cpu_count() or 4
default_workers = 0 if os.name == "nt" else min(8, cpu_workers)
num_workers = int(os.getenv("LEAFWISE_NUM_WORKERS", str(default_workers)))
pin_memory = device.type == "cuda"

if device.type == "cuda":
    train_batch_size = 96
    val_batch_size = 64
else:
    train_batch_size = 16
    val_batch_size = 8

NUM_EPOCHS = int(os.getenv("LEAFWISE_EPOCHS", "12"))
MAX_TRAIN_STEPS = int(os.getenv("LEAFWISE_MAX_TRAIN_STEPS", "0"))
MAX_VAL_STEPS = int(os.getenv("LEAFWISE_MAX_VAL_STEPS", "0"))
RANDOM_SEED = int(os.getenv("LEAFWISE_SEED", "42"))
PATIENCE = int(os.getenv("LEAFWISE_PATIENCE", "3"))

ENABLE_OPTUNA = os.getenv("LEAFWISE_ENABLE_OPTUNA", "0") == "1"
OPTUNA_TRIALS = int(os.getenv("LEAFWISE_OPTUNA_TRIALS", "10"))
OPTUNA_EPOCHS = int(os.getenv("LEAFWISE_OPTUNA_EPOCHS", "4"))
OPTUNA_TIMEOUT = int(os.getenv("LEAFWISE_OPTUNA_TIMEOUT", "0"))

# === Transforms & Dataset ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
dataset = ImageFolder(root=IMAGE_PATH, transform=transform)
num_classes = len(dataset.classes)
class_to_idx = dataset.class_to_idx

# Stratified split improves class balance in validation metrics.
indices = np.arange(len(dataset))
targets = np.array([sample[1] for sample in dataset.samples])
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=targets,
)
train_data = Subset(dataset, train_idx)
val_data = Subset(dataset, val_idx)


def build_loaders(train_bs: int, val_bs: int):
    train_loader_local = DataLoader(
        train_data,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader_local = DataLoader(
        val_data,
        batch_size=val_bs,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader_local, val_loader_local


def build_model():
    try:
        model_local = BeitForImageClassification.from_pretrained(
            'microsoft/beit-base-patch16-224',
            local_files_only=True,
        )
    except Exception:
        model_local = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224')
    model_local.classifier = nn.Linear(model_local.classifier.in_features, num_classes)
    model_local.to(device)
    return model_local

criterion = nn.CrossEntropyLoss()
amp_enabled = device.type == "cuda"
scaler = torch.amp.GradScaler(enabled=amp_enabled)


def train_model(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    epochs,
    patience,
    save_checkpoint=False,
    checkpoint_name='beit_best_checkpoint.pth',
    trial=None,
):
    best_val_loss_local = float('inf')
    best_val_acc_local = 0.0
    best_epoch_local = -1
    counter_local = 0
    csv_log_local = []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        train_steps_ran = 0

        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"), start=1):
            if MAX_TRAIN_STEPS and step > MAX_TRAIN_STEPS:
                break

            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images).logits
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_steps_ran += 1
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / max(train_steps_ran, 1)
        train_accuracy = 100 * correct / max(total, 1)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_steps_ran = 0

        with torch.no_grad():
            for step, (images, labels) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"), start=1):
                if MAX_VAL_STEPS and step > MAX_VAL_STEPS:
                    break

                images = images.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)
                outputs = model(images).logits
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_steps_ran += 1
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= max(val_steps_ran, 1)
        val_accuracy = 100 * val_correct / max(val_total, 1)

        csv_log_local.append([epoch + 1, train_loss, val_loss, train_accuracy, val_accuracy])
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"Train": train_accuracy, "Val": val_accuracy}, epoch)

        print(
            f"Epoch {epoch + 1} ➤ Train Loss: {train_loss:.3f}, Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.3f}, Acc: {val_accuracy:.2f}%"
        )

        scheduler.step(val_loss)
        if val_loss < best_val_loss_local:
            best_val_loss_local = val_loss
            best_val_acc_local = val_accuracy
            best_epoch_local = epoch
            counter_local = 0

            if save_checkpoint:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy,
                    },
                    os.path.join(SAVE_DIR, checkpoint_name),
                )
        else:
            counter_local += 1
            if counter_local >= patience:
                print("⏹ Early stopping triggered.")
                break

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return {
        'best_val_loss': best_val_loss_local,
        'best_val_acc': best_val_acc_local,
        'best_epoch': best_epoch_local,
        'csv_log': csv_log_local,
    }


def run_optuna_search(train_loader, val_loader):
    if optuna is None:
        raise RuntimeError("Optuna is not installed. Install it with: pip install optuna")

    print("🔎 Starting Optuna hyperparameter tuning...")

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-6, 3e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        trial_model = build_model()
        trial_optimizer = optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
        trial_scheduler = ReduceLROnPlateau(trial_optimizer, mode='min', factor=0.5, patience=1)

        result = train_model(
            model=trial_model,
            optimizer=trial_optimizer,
            scheduler=trial_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=OPTUNA_EPOCHS,
            patience=max(2, min(PATIENCE, 3)),
            save_checkpoint=False,
            trial=trial,
        )

        return result['best_val_loss']

    study = optuna.create_study(direction="minimize")
    optimize_kwargs = {"n_trials": OPTUNA_TRIALS}
    if OPTUNA_TIMEOUT > 0:
        optimize_kwargs["timeout"] = OPTUNA_TIMEOUT

    study.optimize(objective, **optimize_kwargs)

    best_params = study.best_params
    best_payload = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": best_params,
        "trials": len(study.trials),
    }

    with open(os.path.join(SAVE_DIR, "optuna_best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    print(f"✅ Optuna best params: {best_params}")
    print(f"✅ Optuna best val loss: {study.best_value:.4f}")
    return best_params


def main():
    train_loader, val_loader = build_loaders(train_batch_size, val_batch_size)
    best_params = None

    if ENABLE_OPTUNA:
        best_params = run_optuna_search(train_loader, val_loader)

    model = build_model()
    lr = best_params["lr"] if best_params else 2e-5
    weight_decay = best_params["weight_decay"] if best_params else 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    print(f"Number of classes: {num_classes}")
    print(f"Class names: {dataset.classes}")
    print(model.classifier)
    print(f"Using lr={lr:.2e}, weight_decay={weight_decay:.2e}")

    train_result = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=NUM_EPOCHS,
        patience=PATIENCE,
        save_checkpoint=True,
    )

    # === Final Evaluation (best checkpoint only) ===
    checkpoint = torch.load(os.path.join(SAVE_DIR, 'beit_best_checkpoint.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    best_all_preds, best_all_labels, best_all_probs = [], [], []
    with torch.no_grad():
        for step, (images, labels) in enumerate(tqdm(val_loader, desc="Best Model [Val Eval]"), start=1):
            if MAX_VAL_STEPS and step > MAX_VAL_STEPS:
                break
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            logits = model(images).logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            best_all_preds.extend(preds.cpu().numpy())
            best_all_labels.extend(labels.cpu().numpy())
            best_all_probs.extend(probs.cpu().numpy())

    best_all_probs = np.asarray(best_all_probs)
    true_bin = label_binarize(best_all_labels, classes=range(num_classes))

    print("\n📊 Classification Report:")
    print(classification_report(best_all_labels, best_all_preds, target_names=dataset.classes, zero_division=0))
    print("\n🔍 Confusion Matrix:")
    print(confusion_matrix(best_all_labels, best_all_preds))

    final_accuracy = accuracy_score(best_all_labels, best_all_preds)
    final_hamming = hamming_loss(best_all_labels, best_all_preds)
    final_mcc = matthews_corrcoef(best_all_labels, best_all_preds)
    final_kappa = cohen_kappa_score(best_all_labels, best_all_preds)

    try:
        final_avg_precision = average_precision_score(true_bin, best_all_probs, average='macro')
    except Exception:
        final_avg_precision = float('nan')

    try:
        final_roc_auc = roc_auc_score(true_bin, best_all_probs, average='macro', multi_class='ovr')
    except Exception:
        final_roc_auc = float('nan')

    print("\n📈 Additional Metrics:")
    print(f"Hamming Loss: {final_hamming:.4f}")
    print(f"Subset Accuracy: {final_accuracy:.4f}")
    print(f"Matthews Corr Coef: {final_mcc:.4f}")
    print(f"Cohen Kappa Score: {final_kappa:.4f}")
    print(f"Avg Precision Score: {final_avg_precision:.4f}")
    print(f"ROC AUC Score: {final_roc_auc:.4f}")

    # === ROC Curve ===
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_bin[:, i], best_all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(true_bin.ravel(), best_all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red',
                    'purple', 'pink', 'brown', 'gray', 'olive'])
    for i, color in zip(range(min(num_classes, 10)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve {dataset.classes[i]} (AUC={roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve (Top 10 classes)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_path = os.path.join(PROJECT_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # === Macro ROC ===
    plt.figure()
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-Average ROC (AUC={roc_auc["macro"]:.2f})', color='navy', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-Average ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    macro_path = os.path.join(PROJECT_DIR, "macro_roc_curve.png")
    plt.savefig(macro_path, dpi=300)
    plt.close()

    print(f"📊 ROC curves saved to: {roc_path}")
    print(f"📈 Macro-average ROC curve saved to: {macro_path}")
    print(f"🏁 Best checkpoint epoch: {checkpoint.get('epoch', -1) + 1}")
    print(f"🏁 Best checkpoint val loss: {checkpoint.get('val_loss', float('nan')):.4f}")
    print(f"🏁 Best checkpoint val acc: {checkpoint.get('val_accuracy', float('nan')):.2f}%")

    # === Logs and Metadata Save ===
    with open(os.path.join(PROJECT_DIR, "epoch_metrics.csv"), "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])
        csv_writer.writerows(train_result['csv_log'])

    with open(os.path.join(SAVE_DIR, "label_encoder.json"), "w") as f:
        json.dump(class_to_idx, f)

    torch.save(model.classifier.state_dict(), os.path.join(SAVE_DIR, "beit_classifier_head.pth"))

    print(f"\n✅ Final Validation Accuracy: {final_accuracy * 100:.2f}%")


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
