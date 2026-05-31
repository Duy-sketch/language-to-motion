"""
Train language model: câu lệnh → action vector [Δα, Δβ, ΔX, ΔY, ΔZ]

Cách dùng:
  1. Điền data vào label_template.csv (càng nhiều càng tốt, tối thiểu ~200 dòng)
  2. Chạy: python train_language_model.py
  3. Model được lưu vào: capsule_language_model/

Cài đặt:
  pip install transformers torch pandas scikit-learn
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import os

# ── Config ────────────────────────────────────────────────────────────────────
BACKBONE    = "vinai/phobert-base"   # PhoBERT tiếng Việt
DATA_FILE   = "label_template.csv"
SAVE_DIR    = "capsule_language_model"
MAX_LEN     = 64
BATCH_SIZE  = 16
EPOCHS      = 30
LR          = 2e-5
ACTION_COLS = ["delta_alpha", "delta_beta", "delta_x", "delta_y", "delta_z"]

# ── Dataset ───────────────────────────────────────────────────────────────────
class CommandDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ── Model ─────────────────────────────────────────────────────────────────────
class CommandToAction(nn.Module):
    def __init__(self, backbone_name, n_actions=5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden_size  = self.encoder.config.hidden_size  # 768 với PhoBERT

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_actions),
        )

    def forward(self, input_ids, attention_mask):
        # Lấy [CLS] token embedding làm đại diện câu
        out        = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed  = out.last_hidden_state[:, 0, :]   # shape: (B, 768)
        action     = self.head(cls_embed)              # shape: (B, 5)
        return action

# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    # Load data
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} samples")

    texts  = df["command"].tolist()
    labels = df[ACTION_COLS].values.tolist()

    # Normalize labels về [-1, 1] để dễ train hơn
    label_arr  = np.array(labels, dtype=np.float32)
    label_max  = np.abs(label_arr).max(axis=0) + 1e-8
    label_norm = label_arr / label_max

    # Lưu label_max để dùng lúc inference
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(f"{SAVE_DIR}/label_max.npy", label_max)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        texts, label_norm.tolist(), test_size=0.15, random_state=42
    )

    # Tokenizer + datasets
    tokenizer    = AutoTokenizer.from_pretrained(BACKBONE)
    train_ds     = CommandDataset(X_train, y_train, tokenizer)
    val_ds       = CommandDataset(X_val,   y_val,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model  = CommandToAction(BACKBONE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_b       = batch["labels"].to(device)

            preds = model(input_ids, attention_mask)
            loss  = loss_fn(preds, labels_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_b       = batch["labels"].to(device)
                preds          = model(input_ids, attention_mask)
                val_loss      += loss_fn(preds, labels_b).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)

        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # Lưu model tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.encoder.save_pretrained(f"{SAVE_DIR}/encoder")
            tokenizer.save_pretrained(f"{SAVE_DIR}/tokenizer")
            torch.save(model.head.state_dict(), f"{SAVE_DIR}/head.pt")
            print(f"  → Saved best model (val_loss={val_loss:.4f})")

    print(f"\nDone! Best val_loss = {best_val_loss:.4f}")
    print(f"Model saved to: {SAVE_DIR}/")

# ── Inference ─────────────────────────────────────────────────────────────────
def load_model():
    """Load model đã train để dùng trong production."""
    tokenizer = AutoTokenizer.from_pretrained(f"{SAVE_DIR}/tokenizer")
    model     = CommandToAction(f"{SAVE_DIR}/encoder")
    model.head.load_state_dict(torch.load(f"{SAVE_DIR}/head.pt", map_location="cpu"))
    model.eval()
    label_max = np.load(f"{SAVE_DIR}/label_max.npy")
    return model, tokenizer, label_max


def predict(text: str, model, tokenizer, label_max) -> dict:
    """
    Nhận câu lệnh → trả về action vector thực tế (mm / độ).

    Ví dụ:
        model, tok, lmax = load_model()
        result = predict("thấy polyp bên trái, sang phải", model, tok, lmax)
        print(result["action"])  # [0.0, 0.0, 4.8, 0.0, 0.1]
    """
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        norm_pred = model(enc["input_ids"], enc["attention_mask"])

    # Denormalize về đơn vị thực
    action = (norm_pred.squeeze().numpy() * label_max).tolist()

    labels = ["Δα(°)", "Δβ(°)", "ΔX(mm)", "ΔY(mm)", "ΔZ(mm)"]
    return {
        "action": action,
        "named":  dict(zip(labels, action)),
        "raw":    text,
    }


if __name__ == "__main__":
    train()
