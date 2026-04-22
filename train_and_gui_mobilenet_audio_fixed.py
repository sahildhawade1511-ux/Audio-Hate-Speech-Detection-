#!/usr/bin/env python3
"""
train_and_gui_mobilenet_audio_fixed.py
MobileNetV2 transfer learning on mel-spectrograms for audio hate/non-hate detection.
Improvements:
 - higher time resolution mel (10ms hop), n_fft=1024
 - longer context window (4s) and wider spectrogram (256 frames)
 - stronger augmentations for hate class
 - oversampling minority class OR class_weights
 - threshold tuning on validation
 - save model in .keras format
 - simple Tkinter GUI + CLI predict
"""

import os
import sys
import math
import json
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# ---------------- USER CONFIG ----------------
SAMPLE_PATH = "/mnt/data/WhatsApp Video 2025-11-19 at 12.08.47_5b578c0d.mp4"  # if exists, used by GUI sample
HATE_DIR = "hate"
NONHATE_DIR = "non_hate"
OUT_MODEL = "mobilenet_audio_model.keras"   # modern Keras format
PRED_CSV = "predictions_test.csv"
LABEL_MAP_JSON = "label_map.json"

SR = 16000
DURATION = 4.0           # 4s window
MAX_SAMPLES = int(SR * DURATION)
N_MELS = 128
SPEC_WIDTH = 256         # wider time axis
SPEC_SHAPE = (N_MELS, SPEC_WIDTH)
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".mp4"}
# ---------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "gui", "predict"], default="train")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--mixup", type=float, default=0.0)
parser.add_argument("--freeze_until", type=int, default=100)
parser.add_argument("--file", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# ---------------- utilities ----------------
def safe_load_audio(path, sr=SR):
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        if y is None:
            return None
        return y.astype(np.float32)
    except Exception:
        try:
            y, orig_sr = sf.read(path, dtype="float32")
            if y is None:
                return None
            if y.ndim > 1:
                y = y.mean(axis=1)
            if orig_sr != sr:
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
            return y.astype(np.float32)
        except Exception:
            return None

def compute_mel_spectrogram(y, sr=SR, n_mels=N_MELS, width=SPEC_WIDTH):
    if y is None or len(y) == 0:
        return np.zeros((n_mels, width), dtype=np.float32)
    hop_length = int(sr * 0.010)   # 10ms hop
    n_fft = 1024
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft, fmin=20, fmax=sr//2)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_min, S_max = S_db.min(), S_db.max()
    S_norm = (S_db - S_min) / (S_max - S_min + 1e-9)
    if S_norm.shape[1] < width:
        pad = width - S_norm.shape[1]
        S_norm = np.pad(S_norm, ((0,0),(0,pad)), mode="constant")
    elif S_norm.shape[1] > width:
        S_norm = S_norm[:, :width]
    return S_norm.astype(np.float32)

def audio_to_3ch_spec(y):
    spec = compute_mel_spectrogram(y)
    return np.repeat(spec[..., np.newaxis], 3, axis=2)

# ---------------- data utilities ----------------
def build_csvs(hate_dir=HATE_DIR, non_dir=NONHATE_DIR, out_dir="prepared_csv", seed=args.seed, split=(0.8,0.1,0.1)):
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    rows = []
    if Path(hate_dir).exists():
        for p in Path(hate_dir).rglob("*"):
            if p.suffix.lower() in AUDIO_EXTS:
                rows.append((str(p.resolve()), 1))
    if Path(non_dir).exists():
        for p in Path(non_dir).rglob("*"):
            if p.suffix.lower() in AUDIO_EXTS:
                rows.append((str(p.resolve()), 0))
    if len(rows) == 0:
        raise SystemExit(f"No audio found in {hate_dir} or {non_dir}. Put audio files there.")
    df = pd.DataFrame(rows, columns=["path","label"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df); t = int(split[0] * n); v = int((split[0]+split[1]) * n)
    train, val, test = df.iloc[:t], df.iloc[t:v], df.iloc[v:]
    train.to_csv(out_path / "train.csv", index=False)
    val.to_csv(out_path / "validation.csv", index=False)
    test.to_csv(out_path / "test.csv", index=False)
    print(f"Wrote CSVs to {out_dir} -> train:{len(train)} val:{len(val)} test:{len(test)}")
    return train, val, test

def oversample_minority(df, target_col="label"):
    counts = df[target_col].value_counts()
    maxc = counts.max()
    parts = []
    for c, cnt in counts.items():
        part = df[df[target_col]==c]
        if cnt < maxc:
            reps = int(np.ceil(maxc / cnt))
            new = pd.concat([part]*reps, ignore_index=True).sample(n=maxc, random_state=args.seed)
        else:
            new = part.sample(n=maxc, random_state=args.seed)
        parts.append(new)
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=args.seed)
    return out

# ---------------- generator ----------------
class AudioSequence(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=8, shuffle=True, augment=False, mixup_alpha=0.0, duration=DURATION):
        self.df = df.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.augment = bool(augment)
        self.mixup_alpha = float(mixup_alpha)
        self.duration = float(duration)
        self.max_samples = int(SR * self.duration)
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        bs = len(batch_idx)
        X = np.zeros((bs, SPEC_SHAPE[0], SPEC_SHAPE[1], 3), dtype=np.float32)
        y = np.zeros((bs,), dtype=np.int32)
        for i, j in enumerate(batch_idx):
            row = self.df.iloc[j]
            path = str(row["path"])
            label = int(row["label"])
            y_wave = safe_load_audio(path)
            if y_wave is None:
                y_wave = np.zeros(self.max_samples, dtype=np.float32)
            if len(y_wave) > self.max_samples:
                start = random.randint(0, len(y_wave) - self.max_samples)
                y_crop = y_wave[start:start + self.max_samples]
            else:
                y_crop = np.pad(y_wave, (0, max(0, self.max_samples - len(y_wave))))
            # stronger augment for hate class
            if self.augment:
                aug_p = 0.6 if label==1 else 0.35
                if random.random() < aug_p * 0.5:
                    try:
                        n_steps = random.uniform(-2.0, 2.0)
                        y_crop = librosa.effects.pitch_shift(y_crop, sr=SR, n_steps=n_steps)
                    except Exception:
                        pass
                if random.random() < aug_p * 0.6:
                    rms = np.sqrt(np.mean(y_crop**2) + 1e-9)
                    noise = np.random.normal(0, rms * random.uniform(0.03, 0.12), size=y_crop.shape).astype(np.float32)
                    y_crop = (y_crop + noise).astype(np.float32)
                if random.random() < aug_p * 0.25:
                    try:
                        rate = random.uniform(0.9, 1.1)
                        y_new = librosa.effects.time_stretch(y_crop, rate)
                        if len(y_new) > self.max_samples:
                            y_crop = y_new[:self.max_samples]
                        else:
                            y_crop = np.pad(y_new, (0, max(0, self.max_samples - len(y_new))))
                    except Exception:
                        pass
            spec3 = audio_to_3ch_spec(y_crop)
            X[i] = spec3
            y[i] = label
        # mixup support
        if self.mixup_alpha and self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=(bs,))
            perm = np.random.permutation(bs)
            X2 = X[perm]; y2 = y[perm]
            X_mix = X * lam.reshape((-1,1,1,1)) + X2 * (1 - lam).reshape((-1,1,1,1))
            y_one = tf.keras.utils.to_categorical(y, num_classes=2)
            y2_one = tf.keras.utils.to_categorical(y2, num_classes=2)
            y_mix = y_one * lam.reshape((-1,1)) + y2_one * (1 - lam).reshape((-1,1))
            return X_mix.astype(np.float32), y_mix.astype(np.float32)
        return X, y

# ---------------- model ----------------
def make_mobilenet_transfer(input_shape=(SPEC_SHAPE[0], SPEC_SHAPE[1], 3), lr=1e-5, freeze_until=100, use_categorical_loss=False):
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg")
    for layer in base.layers[:freeze_until]:
        layer.trainable = False
    for layer in base.layers[freeze_until:]:
        layer.trainable = True
    x = base.output
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)   # stronger dropout
    out = layers.Dense(2, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=out)
    loss = tf.keras.losses.CategoricalCrossentropy() if use_categorical_loss else "sparse_categorical_crossentropy"
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model

# ---------------- sliding predict + threshold tuning ----------------
def sliding_predict_probs(model, path, window_sec=DURATION, hop_sec=None):
    if hop_sec is None:
        hop_sec = max(0.5, window_sec / 2.0)
    y = safe_load_audio(path)
    if y is None or len(y) == 0:
        return None
    win = int(window_sec * SR); hop = int(hop_sec * SR)
    probs_list = []
    if len(y) <= win:
        X = np.repeat(compute_mel_spectrogram(y)[..., np.newaxis], 3, axis=2)[np.newaxis,...]
        p = model.predict(X, verbose=0)[0]
        return p
    for start in range(0, max(1, len(y) - win + 1), hop):
        chunk = y[start:start+win]
        X = np.repeat(compute_mel_spectrogram(chunk)[..., np.newaxis], 3, axis=2)[np.newaxis,...]
        p = model.predict(X, verbose=0)[0]
        probs_list.append(p)
    if len(probs_list) == 0:
        return None
    avg = np.mean(np.stack(probs_list, axis=0), axis=0)
    return avg

def find_best_threshold(model, val_df):
    probs = []
    trues = []
    for _, row in val_df.iterrows():
        p = sliding_predict_probs(model, row['path'])
        probs.append(float(p[1]) if p is not None else 0.0)
        trues.append(int(row['label']))
    best_t = 0.5; best_f1 = 0.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = [1 if pr >= t else 0 for pr in probs]
        f1 = f1_score(trues, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_t = t
    print("Best threshold on val:", best_t, "f1:", best_f1)
    return best_t

# ---------------- training & evaluation ----------------
def train_and_save(smoke=False, epochs=20, batch_size=8, mixup_alpha=0.0, lr=1e-5, freeze_until=100):
    print("Building CSVs...")
    train_df, val_df, test_df = build_csvs()
    if smoke:
        train_df = train_df.sample(n=min(len(train_df), 200), random_state=args.seed)
        val_df = val_df.sample(n=min(len(val_df), 40), random_state=args.seed)
        test_df = test_df.sample(n=min(len(test_df), 40), random_state=args.seed)
        print("SMOKE test -> smaller slices")
    # oversample minority (preferred) to balance train
    train_df_bal = oversample_minority(train_df)
    # compute class weights as fallback
    cw = compute_class_weight("balanced", classes=np.unique(train_df['label']), y=train_df['label'])
    class_weights = {i: float(w) for i,w in enumerate(cw)}
    print("Class weights:", class_weights)
    train_gen = AudioSequence(train_df_bal, batch_size=batch_size, shuffle=True, augment=True, mixup_alpha=mixup_alpha)
    val_gen = AudioSequence(val_df, batch_size=batch_size, shuffle=False, augment=False, mixup_alpha=0.0)
    use_categorical = True if mixup_alpha and mixup_alpha > 0 else False
    model = make_mobilenet_transfer(lr=lr, freeze_until=freeze_until, use_categorical_loss=use_categorical)
    model.summary()
    # sanity check batch shapes
    xb, yb = train_gen[0]
    print("DEBUG train batch shapes:", xb.shape, yb.shape)
    cb = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(OUT_MODEL, monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
    ]
    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=cb,
        class_weight=(None if use_categorical else class_weights),
        verbose=2
    )
    print("Training complete. Saving model and label map...")
    model.save(OUT_MODEL)
    with open(LABEL_MAP_JSON, "w") as f:
        json.dump({"0":"non-hate", "1":"hate"}, f)
    # threshold tuning
    print("Finding best threshold on validation set...")
    best_t = find_best_threshold(model, val_df)
    # sliding-window evaluation on test
    print("Running sliding-window evaluation on test set...")
    results = []
    for _, row in test_df.reset_index(drop=True).iterrows():
        p = row["path"]; true = int(row["label"])
        probs = sliding_predict_probs(model, p)
        if probs is None:
            pred = 0; prob_hate = 0.0
        else:
            prob_hate = float(probs[1]) if len(probs)>1 else 0.0
            pred = 1 if prob_hate >= best_t else 0
        results.append({"path": p, "true": true, "pred": pred, "prob_hate": prob_hate})
    out_df = pd.DataFrame(results)
    out_df.to_csv(PRED_CSV, index=False)
    print("Saved predictions to", PRED_CSV)
    y_true = out_df["true"].values; y_pred = out_df["pred"].values
    acc = accuracy_score(y_true, y_pred); p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    print(f"Test metrics -> accuracy: {acc:.4f} precision: {p:.4f} recall: {r:.4f} f1: {f1:.4f}")
    return model, out_df

# ---------------- GUI ----------------
def run_gui():
    if not Path(OUT_MODEL).exists():
        messagebox.showerror("Model missing", f"Train first: {OUT_MODEL} not found")
        return
    model = tf.keras.models.load_model(OUT_MODEL)
    with open(LABEL_MAP_JSON, "r") as f:
        label_map = json.load(f)
    root = tk.Tk()
    root.title("Harmful Audio Detector — MobileNet (fixed)")
    root.geometry("780x260")
    root.resizable(False, False)
    frm = ttk.Frame(root, padding=12); frm.pack(fill="both", expand=True)
    ttk.Label(frm, text=f"Model: {OUT_MODEL}", font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", columnspan=3)
    file_lbl = ttk.Label(frm, text="No file selected", width=80); file_lbl.grid(row=1, column=0, sticky="w")
    def choose_file():
        p = filedialog.askopenfilename(title="Choose audio file", filetypes=[("Audio","*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.mp4")])
        if p:
            file_lbl.config(text=p); result_lbl.config(text=""); prob_lbl.config(text="")
    ttk.Button(frm, text="Choose File", command=choose_file).grid(row=1, column=1, padx=8)
    def predict_cmd():
        path = file_lbl.cget("text")
        if not path or path == "No file selected":
            messagebox.showinfo("No file", "Choose a file first."); return
        try:
            probs = sliding_predict_probs(model, path)
            if probs is None:
                messagebox.showerror("Error", "Could not load audio or compute features."); return
            pred = int(np.argmax(probs)); prob_hate = float(probs[1]) if len(probs)>1 else 0.0
            label = label_map.get(str(pred), str(pred)).upper()
            result_lbl.config(text=label, foreground=("red" if pred==1 else "green"), font=("Segoe UI", 20, "bold"))
            prob_lbl.config(text=f"Prob(hate)={prob_hate:.3f}  Prob(non-hate)={probs[0]:.3f}")
        except Exception as e:
            messagebox.showerror("Prediction error", str(e))
    ttk.Button(frm, text="Predict", command=predict_cmd).grid(row=1, column=2, padx=8)
    result_lbl = ttk.Label(frm, text="", font=("Segoe UI", 20)); result_lbl.grid(row=2, column=0, columnspan=3, pady=(12,0))
    prob_lbl = ttk.Label(frm, text="", font=("Segoe UI", 10)); prob_lbl.grid(row=3, column=0, columnspan=3)
    def load_sample():
        if Path(SAMPLE_PATH).exists():
            file_lbl.config(text=SAMPLE_PATH)
        else:
            messagebox.showinfo("Sample missing", f"Sample not found: {SAMPLE_PATH}")
    ttk.Button(frm, text="Use sample file (if present)", command=load_sample).grid(row=4, column=0, sticky="w", pady=(10,0))
    root.mainloop()

# ---------------- CLI predict ----------------
def predict_file_cli(model_path, file_path):
    if not Path(model_path).exists():
        raise SystemExit("Model not found, train first.")
    model = tf.keras.models.load_model(model_path)
    probs = sliding_predict_probs(model, file_path)
    if probs is None:
        print("Could not load audio or compute features."); return
    pred = int(np.argmax(probs))
    print("Pred:", pred, "Prob(hate)=", float(probs[1]) if len(probs)>1 else 0.0)

# ---------------- main ----------------
if __name__ == "__main__":
    if args.mode == "train":
        train_and_save(smoke=args.smoke, epochs=args.epochs, batch_size=args.batch, mixup_alpha=args.mixup, lr=args.lr, freeze_until=args.freeze_until)
    elif args.mode == "gui":
        run_gui()
    else:
        if not args.file:
            print("Use --file to specify audio file"); sys.exit(1)
        predict_file_cli(OUT_MODEL, args.file)
