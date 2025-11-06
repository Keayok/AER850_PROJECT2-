# -*- coding: utf-8 -*-
"""


@author: quich
"""

# ============================================================
#  AER850 Project 2  |  Steps 1–4  (Train only — NO test here)
#  - Strong augments, LeakyReLU, AMP, tqdm, early stopping
#  - Step 3: fast tuning @256px + stratified subset
#  - Step 4: full training @500px from-scratch (no test)
#  - Saves:
#      best_resnet.pt, best_customcnn.pt
#      hist_resnet.pt, hist_customcnn.pt
# ============================================================

# -----------------------------
# STEP 1: DATA PREPARATION
# -----------------------------
from pathlib import Path
import json, copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

# ======= EDIT THIS PATH =======
ROOT_DIR = Path(r"C:\Users\quich\Downloads\AER850\AER850_PROJECT2\Project 2 Data")
# ==============================

IMG_SIZE     = 500
BATCH_SIZE   = 32
NUM_WORKERS  = 0  # keep 0 for Windows/Spyder stability

TRAIN_DIR = ROOT_DIR / "data" / "train"
VAL_DIR   = ROOT_DIR / "data" / "valid"
TEST_DIR  = ROOT_DIR / "data" / "test"     # used only in Step 5

# Full-res training augments (Keras-like)
# FIX: RandomErasing must come AFTER ToTensor() (it expects a Tensor).
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=15, translate=(0.05,0.05), scale=(0.9,1.1), shear=8),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.25, value='random'),
])

# Val is clean
eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = ImageFolder(str(TRAIN_DIR), transform=train_tfms)
val_ds   = ImageFolder(str(VAL_DIR),   transform=eval_tfms)
test_ds  = ImageFolder(str(TEST_DIR),  transform=eval_tfms)  # not used here; Step 5 only

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print("Classes:", train_ds.classes)
print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")

# -----------------------------
# STEP 2: ARCH DEFINITIONS
# -----------------------------
import torch.nn as nn
import torchvision.models as models
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Utility: Kaiming init for LeakyReLU layers
def kaiming_init(module, a=0.1):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=a, mode='fan_in', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

NUM_CLASSES = len(train_ds.classes)

# ResNet-18 transfer learning backbone + custom head
def build_resnet18(dropout_p: float):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 512),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout_p),
        nn.Linear(512, NUM_CLASSES)
    )
    m.fc.apply(lambda mod: kaiming_init(mod, a=0.1))
    # freeze backbone initially (head warm-up)
    for name, p in m.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = False
    return m.to(device).to(memory_format=torch.channels_last)

# ---- Custom CNN (improved) ----
# Depthwise+Pointwise convs + lightweight residual + SE attention
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, max(8, c//r)), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(max(8, c//r), c), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w

class DWConv(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.bn(x); return self.act(x)

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = DWConv(c, c)
        self.conv2 = DWConv(c, c)
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class CustomCNN_SE(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        act = lambda: nn.LeakyReLU(0.1, inplace=True)
        self.stem  = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), act())
        self.stage1= nn.Sequential(DWConv(32, 64,  stride=2), SEBlock(64),  ResidualBlock(64))   # 500->250
        self.stage2= nn.Sequential(DWConv(64, 128, stride=2), SEBlock(128), ResidualBlock(128))  # 250->125
        self.stage3= nn.Sequential(DWConv(128,256, stride=2), SEBlock(256), ResidualBlock(256))  # 125->63
        self.stage4= nn.Sequential(DWConv(256,384, stride=2), SEBlock(384), ResidualBlock(384))  # 63->32
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.head  = nn.Sequential(nn.Flatten(), nn.Linear(384, 512), act(), nn.Dropout(0.45), nn.Linear(512, num_classes))
        self.apply(lambda m: kaiming_init(m, a=0.1))
    def forward(self, x):
        x = self.stem(x); x=self.stage1(x); x=self.stage2(x); x=self.stage3(x); x=self.stage4(x)
        x = self.pool(x); x=self.head(x); return x

def build_customcnn():
    return CustomCNN_SE(NUM_CLASSES).to(device).to(memory_format=torch.channels_last)

print("Step 2 models ready.")

# -----------------------------
# STEP 3: FAST JOINT TUNING  (256px + stratified subset)
# -----------------------------
from collections import defaultdict
from tqdm.auto import tqdm

# perf knobs
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

TRIAL_EPOCHS    = 5
EARLY_PATIENCE  = 2
WEIGHT_DECAY    = 1e-4
MOMENTUM_SGD    = 0.9

IMG_SIZE_TRIAL  = 256
TRIAL_BATCH     = 64

# FIX: RandomErasing must come AFTER ToTensor()
trial_train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE_TRIAL, IMG_SIZE_TRIAL)),
    transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=15, translate=(0.05,0.05), scale=(0.9,1.1), shear=8),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.25, value='random'),
])
trial_val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE_TRIAL, IMG_SIZE_TRIAL)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

trial_train_full = ImageFolder(str(TRAIN_DIR), transform=trial_train_tfms)
trial_val_full   = ImageFolder(str(VAL_DIR),   transform=trial_val_tfms)

def stratified_subset(dataset: ImageFolder, max_per_class=150, seed=42):
    rng = np.random.default_rng(seed)
    by_cls = defaultdict(list)
    for idx, (_, y) in enumerate(dataset.samples):
        by_cls[y].append(idx)
    keep = []
    for y, idxs in by_cls.items():
        idxs = np.array(idxs)
        if len(idxs) > max_per_class:
            idxs = rng.choice(idxs, size=max_per_class, replace=False)
        keep.extend(idxs.tolist())
    keep.sort()
    return Subset(dataset, keep)

trial_train_ds = stratified_subset(trial_train_full, max_per_class=150)
trial_val_ds   = trial_val_full

TRAIN_LOADER = DataLoader(trial_train_ds, batch_size=TRIAL_BATCH, shuffle=True,
                          num_workers=0, pin_memory=True)
VAL_LOADER   = DataLoader(trial_val_ds,   batch_size=TRIAL_BATCH, shuffle=False,
                          num_workers=0, pin_memory=True)

def eval_once(model, loader, criterion):
    model.eval()
    tot_loss = tot_ok = tot_n = 0
    amp = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available())
    with amp, torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).to(memory_format=torch.channels_last)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            tot_loss += loss.item() * xb.size(0)
            tot_ok   += (logits.argmax(1) == yb).sum().item()
            tot_n    += xb.size(0)
    return tot_loss / tot_n, tot_ok / tot_n

def count_params(m):
    tot = sum(p.numel() for p in m.parameters())
    trn = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return tot, trn

def build_model(arch, dropout_p):
    if arch == "resnet18":
        m = build_resnet18(dropout_p)
    elif arch == "customcnn":
        m = build_customcnn()
    else:
        raise ValueError("arch must be 'resnet18' or 'customcnn'")
    return m

def train_quick_trial(arch, dropout_p, opt_name, lr, epochs=TRIAL_EPOCHS):
    model = build_model(arch, dropout_p)
    tot, trn = count_params(model)
    print(f"[build] {arch} drop={dropout_p} | total={tot/1e6:.2f}M trainable={trn/1e6:.2f}M")

    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=MOMENTUM_SGD, weight_decay=WEIGHT_DECAY, nesterov=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())  # FIX: new AMP API

    best_val_acc, best_state, no_improve = 0.0, None, 0
    hist = defaultdict(list)

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = tr_ok = tr_n = 0
        pbar = tqdm(total=len(TRAIN_LOADER), leave=False,
                    desc=f"{arch} d={dropout_p} {opt_name} lr={lr:.1e} ep {ep}/{epochs}")
        for xb, yb in TRAIN_LOADER:
            xb = xb.to(device).to(memory_format=torch.channels_last)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * xb.size(0)
            tr_ok   += (logits.argmax(1) == yb).sum().item()
            tr_n    += xb.size(0)
            pbar.update(1)
        pbar.close()

        tr_loss /= tr_n
        tr_acc   = tr_ok / tr_n
        val_loss, val_acc = eval_once(model, VAL_LOADER, criterion)

        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(val_loss);   hist["val_acc"].append(val_acc)

        print(f"[trial {arch}] Ep{ep:02d}/{epochs} | train {tr_loss:.4f}/{tr_acc:.3f}  val {val_loss:.4f}/{val_acc:.3f}")

        if val_acc > best_val_acc + 1e-4:
            best_val_acc, best_state, no_improve = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= EARLY_PATIENCE:
                print(f"  ↳ early stop (no val acc improvement {EARLY_PATIENCE} epochs)")
                break
    return best_val_acc, hist, best_state

# Small grid across both models
trial_space = {
    "arch":    ["resnet18", "customcnn"],
    "dropout": [0.3, 0.5],       # (ignored by customcnn)
    "opt":     ["adam", "sgd"],
    "lr":      [1e-3, 7e-4, 3e-4],
}

leaderboard = []
for arch in trial_space["arch"]:
    for d in trial_space["dropout"]:
        if arch == "customcnn" and d != 0.3:  # dropout not used in customcnn head; skip dup
            continue
        for optn in trial_space["opt"]:
            for lr in trial_space["lr"]:
                best_acc, hist, state = train_quick_trial(arch, d, optn, lr)
                cfg = {"arch": arch, "dropout": d, "opt": optn, "lr": lr}
                leaderboard.append((best_acc, cfg, state))
                print(f"--> DONE | best val_acc={best_acc:.4f} | cfg={cfg}")

leaderboard.sort(key=lambda x: x[0], reverse=True)
print("\n===== LEADERBOARD (top → bottom) =====")
for i,(acc,cfg,_) in enumerate(leaderboard,1):
    print(f"{i:2d}. {acc:.4f} | {cfg}")

def best_for_arch(arch):
    for acc,cfg,state in leaderboard:
        if cfg["arch"]==arch: return acc,cfg,state
    return None,None,None

best_resnet_acc,best_resnet_cfg,best_resnet_state = best_for_arch("resnet18")
best_custom_acc,best_custom_cfg,best_custom_state = best_for_arch("customcnn")

print("\nSelected ResNet:",best_resnet_cfg,"≈",f"{best_resnet_acc:.4f}")
print("Selected CustomCNN:",best_custom_cfg,"≈",f"{best_custom_acc:.4f}")

# -----------------------------
# STEP 4: FULL TRAIN @ 500×500 (no test here)
# -----------------------------
from tqdm.auto import tqdm

FULL_EPOCHS     = 45           # longer for from-scratch customcnn
WARMUP_EPOCHS   = 3            # for resnet head-only warmup
EARLY_PATIENCE4 = 4            # early stop on val loss
MOMENTUM_SGD    = 0.9
WEIGHT_DECAY    = 1e-4

def make_optimizer(params, opt_name, lr):
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)
    else:
        return torch.optim.SGD(params, lr=lr, momentum=MOMENTUM_SGD, weight_decay=WEIGHT_DECAY, nesterov=True)

def evaluate(model, loader, criterion):
    model.eval()
    tot_loss = tot_ok = tot_n = 0
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()), torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).to(memory_format=torch.channels_last)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            tot_loss += loss.item() * xb.size(0)
            tot_ok   += (logits.argmax(1) == yb).sum().item()
            tot_n    += xb.size(0)
    return tot_loss / tot_n, tot_ok / tot_n

def unfreeze_resnet_last_block(model):
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("layer4.") or name.startswith("fc.")
    return [p for p in model.parameters() if p.requires_grad]

def rebuild_for_full(cfg):
    if cfg["arch"] == "resnet18":
        model = build_resnet18(cfg["dropout"])
    else:
        model = build_customcnn()
    # NOTE: retrain from scratch — do NOT load trial weights to avoid leakage
    return model

def train_full(arch_name, cfg, save_path, force_batch=None):
    batch_size = force_batch if force_batch else BATCH_SIZE
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True)
    vl_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

    model = rebuild_for_full(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(params, cfg["opt"], cfg["lr"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FULL_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_loss, best_val_acc = float("inf"), 0.0
    best_state = None
    hist = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[], "lr":[]}
    no_improve = 0

    for ep in range(1, FULL_EPOCHS+1):
        if arch_name == "resnet18" and ep == WARMUP_EPOCHS + 1:
            params = unfreeze_resnet_last_block(model)
            base_lr = cfg["lr"] * 0.5
            optimizer = make_optimizer(params, cfg["opt"], base_lr)

        model.train()
        tr_loss = tr_ok = tr_n = 0
        pbar = tqdm(total=len(tr_loader), leave=False, desc=f"{arch_name} | ep {ep}/{FULL_EPOCHS}")
        for xb, yb in tr_loader:
            xb = xb.to(device).to(memory_format=torch.channels_last)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item() * xb.size(0)
            tr_ok   += (logits.argmax(1) == yb).sum().item()
            tr_n    += xb.size(0)
            pbar.update(1)
        pbar.close()

        tr_loss /= tr_n
        tr_acc   = tr_ok / tr_n

        val_loss, val_acc = evaluate(model, vl_loader, criterion)
        scheduler.step()

        last_lr = optimizer.param_groups[0]["lr"]
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(val_loss);   hist["val_acc"].append(val_acc)
        hist["lr"].append(last_lr)

        print(f"[{arch_name}] Epoch {ep:02d}/{FULL_EPOCHS} | "
              f"train {tr_loss:.4f}/{tr_acc:.3f}  val {val_loss:.4f}/{val_acc:.3f}  lr {last_lr:.2e}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss, best_val_acc = val_loss, val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({"cfg": cfg, "classes": train_ds.classes,
                        "model_state": best_state}, save_path)
            torch.save(hist, save_path.replace(".pt", "_hist.pt"))
        else:
            no_improve += 1
            if no_improve >= EARLY_PATIENCE4:
                print(f"  ↳ early stop (no val loss improvement for {EARLY_PATIENCE4} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_val_loss, best_val_acc

# Train both using the tuned configs
print("\n=== Step 4: Train ResNet-18 @500px ===")
resnet_model, resnet_hist, resnet_vloss, resnet_vacc = train_full(
    "resnet18", best_resnet_cfg, save_path="best_resnet.pt", force_batch=BATCH_SIZE
)

print("\n=== Step 4: Train CustomCNN @500px (smaller batch) ===")
custom_batch = 16
best_custom_cfg["lr"] = min(best_custom_cfg["lr"], 7e-4)
custom_model, custom_hist, custom_vloss, custom_vacc = train_full(
    "customcnn", best_custom_cfg, save_path="best_customcnn.pt", force_batch=custom_batch
)

torch.save(resnet_hist, "hist_resnet.pt")
torch.save(custom_hist, "hist_customcnn.pt")

print("\nStep 4 complete — checkpoints and histories saved:")
print("  - best_resnet.pt, hist_resnet.pt")
print("  - best_customcnn.pt, hist_customcnn.pt")


import matplotlib.pyplot as plt
# Figure A: training curves
plt.figure(figsize=(11,5))
plt.subplot(1,2,1)
plt.plot(resnet_hist["train_loss"], label="ResNet train")
plt.plot(resnet_hist["val_loss"],   label="ResNet val")
plt.plot(custom_hist["train_loss"], label="Custom train", linestyle="--")
plt.plot(custom_hist["val_loss"],   label="Custom val", linestyle="--")
plt.title("Loss vs Epoch"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.subplot(1,2,2)
plt.plot(resnet_hist["train_acc"], label="ResNet train")
plt.plot(resnet_hist["val_acc"],   label="ResNet val")
plt.plot(custom_hist["train_acc"], label="Custom train", linestyle="--")
plt.plot(custom_hist["val_acc"],   label="Custom val", linestyle="--")
plt.title("Accuracy vs Epoch"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

plt.tight_layout()
plt.savefig("figure_model_performance.png", dpi=160)
print("Saved: figure_model_performance.png")
