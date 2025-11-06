# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 20:08:34 2025

@author: quich
"""

# ============================================================
#  AER850 Project 2  |  Step 5 (Evaluation & Figures)
# ============================================================
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import torchvision.models as models

# ======= EDIT THIS PATH =======
ROOT_DIR = Path(r"C:\Users\quich\Downloads\AER850\AER850_PROJECT2\Project 2 Data")
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DIR = ROOT_DIR / "data" / "test"
IMG_SIZE = 500

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

test_ds = ImageFolder(str(TEST_DIR), transform=eval_tfms)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

def kaiming_init(module, a=0.1):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=a, mode='fan_in', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# --- Custom CNN (same as train file) ---
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
        self.stage1= nn.Sequential(DWConv(32, 64,  stride=2), SEBlock(64),  ResidualBlock(64))
        self.stage2= nn.Sequential(DWConv(64, 128, stride=2), SEBlock(128), ResidualBlock(128))
        self.stage3= nn.Sequential(DWConv(128,256, stride=2), SEBlock(256), ResidualBlock(256))
        self.stage4= nn.Sequential(DWConv(256,384, stride=2), SEBlock(384), ResidualBlock(384))
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.head  = nn.Sequential(nn.Flatten(), nn.Linear(384, 512), act(), nn.Dropout(0.45), nn.Linear(512, num_classes))
        self.apply(lambda m: kaiming_init(m, a=0.1))
    def forward(self, x):
        x = self.stem(x); x=self.stage1(x); x=self.stage2(x); x=self.stage3(x); x=self.stage4(x)
        x = self.pool(x); x=self.head(x); return x

def build_resnet18(num_classes, dropout_p):
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 512),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout_p),
        nn.Linear(512, num_classes)
    )
    return m

# Load checkpoints & histories
res_ckpt = torch.load("best_resnet.pt", map_location="cpu")
cu_ckpt  = torch.load("best_customcnn.pt", map_location="cpu")
classes = res_ckpt.get("classes", test_ds.classes)
num_classes = len(classes)

resnet = build_resnet18(num_classes, dropout_p=res_ckpt["cfg"].get("dropout", 0.3))
resnet.load_state_dict(res_ckpt["model_state"])
resnet = resnet.to(device).eval()

custom = CustomCNN_SE(num_classes)
custom.load_state_dict(cu_ckpt["model_state"])
custom = custom.to(device).eval()


@torch.no_grad()
def evaluate(model, loader):
    crit = nn.CrossEntropyLoss()
    tot_loss = tot_ok = tot_n = 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        logits = model(xb)
        loss = crit(logits, yb)
        tot_loss += loss.item() * xb.size(0)
        tot_ok   += (logits.argmax(1) == yb).sum().item()
        tot_n    += xb.size(0)
    return tot_loss/tot_n, tot_ok/tot_n

res_loss, res_acc = evaluate(resnet, test_loader)
cus_loss, cus_acc = evaluate(custom, test_loader)
print(f"[TEST] ResNet-18: loss={res_loss:.4f}  acc={res_acc:.3f}")
print(f"[TEST] CustomCNN:  loss={cus_loss:.4f}  acc={cus_acc:.3f}")


# Figure B: required three test image predictions
SAMPLES = [
    TEST_DIR / "crack"        / "test_crack.jpg",
    TEST_DIR / "missing-head" / "test_missinghead.jpg",
    TEST_DIR / "paint-off"    / "test_paintoff.jpg",
]

inv_norm = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def show_preds_grid(model, name, paths, cols=3):
    """
    For each image:
      - show the image (de-normalized)
      - show TRUE label
      - show PRED label + confidence
      - overlay ALL class probabilities (%) sorted high→low
    """
    model.eval()
    rows = int(np.ceil(len(paths) / cols))
    plt.figure(figsize=(4*cols, 4*rows))

    for i, p in enumerate(paths, 1):
        # load + preprocess
        img = Image.open(p).convert("RGB")
        x = eval_tfms(img).unsqueeze(0).to(device)

        # forward
        with torch.no_grad():
            logits = model(x)
            probs  = logits.softmax(dim=1).squeeze(0).cpu().numpy()

        # predicted
        pred_idx = int(np.argmax(probs))
        pred_lbl = classes[pred_idx]
        pred_conf = probs[pred_idx] * 100

        # true label from parent folder name (matches ImageFolder class names)
        true_name = p.parent.name
        try:
            true_idx = classes.index(true_name)
        except ValueError:
            # fallback: if folder naming doesn't match classes ordering
            # (shouldn't happen with ImageFolder), default to -1
            true_idx = -1
        correct = (pred_idx == true_idx)
        true_lbl = true_name if true_idx >= 0 else "(unknown)"

        # prepare image for display
        xshow = inv_norm(x[0].cpu()).clamp(0,1).permute(1,2,0).numpy()

        # subplot
        ax = plt.subplot(rows, cols, i)
        ax.imshow(xshow)
        ax.axis("off")

        # title: TRUE vs PRED (color-coded)
        title_color = "tab:green" if correct else "tab:red"
        ax.set_title(
            f"{name}\nTRUE: {true_lbl} | PRED: {pred_lbl} ({pred_conf:.1f}%)",
            color=title_color, fontsize=10
        )

        # overlay ALL class probabilities (%), sorted
        order = np.argsort(-probs)  # descending
        lines = [f"{classes[j]:<12}: {probs[j]*100:5.1f}%" for j in order]
        text_block = "\n".join(lines)

        ax.text(
            0.01, 0.01, text_block,
            transform=ax.transAxes,
            fontsize=9,
            family="monospace",
            va="bottom", ha="left",
            color="white",
            bbox=dict(facecolor="black", alpha=0.55, boxstyle="round,pad=0.3")
        )

    fn = f"figure_test_{name.lower()}.png"
    plt.tight_layout()
    plt.savefig(fn, dpi=160)
    print(f"Saved: {fn}")

# --- create the two required figures (no training curves here) ---
show_preds_grid(resnet, "ResNet18", SAMPLES)
show_preds_grid(custom, "CustomCNN", SAMPLES)

print("\nStep 5 complete — exported:")
print("  - figure_test_resnet18.png")
print("  - figure_test_customcnn.png")