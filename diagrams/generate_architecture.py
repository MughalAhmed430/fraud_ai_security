"""
diagrams/generate_architecture.py
Generates the system architecture diagram as a PNG using matplotlib.
Run:  python diagrams/generate_architecture.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

os.makedirs("diagrams", exist_ok=True)

def draw_box(ax, x, y, w, h, lines, bg_color, text_color="white", fontsize=9):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=bg_color, edgecolor="#ffffff33", linewidth=1.5,
                          zorder=2)
    ax.add_patch(box)
    text = "\n".join(lines)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold", zorder=3, linespacing=1.5)

def arrow(ax, x1, y1, x2, y2, color="#aaaaaa"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0), zorder=4)

fig, ax = plt.subplots(figsize=(17, 10))
ax.set_xlim(0, 17); ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("#0f0f23")
ax.set_facecolor("#0f0f23")

# Background panels
for (x, y, w, h, label, c) in [
    (8.5, 8.1, 16.6, 1.6, "DATA LAYER", "#1a2744"),
    (8.5, 6.3, 16.6, 1.4, "MODEL LAYER", "#2a1a44"),
    (8.5, 4.8, 16.6, 1.2, "METRICS", "#1a3322"),
    (8.5, 3.1, 16.6, 1.8, "ADVERSARIAL ATTACKS (IBM ART)", "#3a1a1a"),
    (8.5, 1.3, 16.6, 1.8, "DEFENSES", "#1a3322"),
]:
    panel = FancyBboxPatch((x-w/2, y-h/2), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=c, edgecolor="#ffffff22", linewidth=1,
                            alpha=0.6, zorder=0)
    ax.add_patch(panel)
    ax.text(0.6, y + h/2 - 0.15, label, fontsize=7, color="#ffffff55",
            fontweight="bold", zorder=1)

# Title
ax.text(8.5, 9.55, "Financial AI Security  —  System Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold", color="white")
ax.text(8.5, 9.2, "Evasion Attacks on Fraud Detection Systems",
        ha="center", va="center", fontsize=10, color="#aaaacc")

# Row 1 — Data
draw_box(ax, 2.5, 8.1, 3.6, 1.0, ["Kaggle Dataset", "284,807 Transactions"], "#1565C0")
draw_box(ax, 6.8, 8.1, 3.6, 1.0, ["Preprocessing", "SMOTE + Scaling"], "#1976D2")
draw_box(ax, 11.0, 8.1, 3.6, 1.0, ["Train / Test Split", "80% / 20%  |  Stratified"], "#1565C0")
draw_box(ax, 15.0, 8.1, 2.8, 1.0, ["Imbalance Fix", "0.17% fraud -> 50/50"], "#0D47A1")

arrow(ax, 4.3, 8.1, 5.0, 8.1); arrow(ax, 8.6, 8.1, 9.2, 8.1); arrow(ax, 12.8, 8.1, 13.6, 8.1)

# Row 2 — Model
draw_box(ax, 8.5, 6.3, 10.0, 1.1,
         ["Neural Network  —  Fraud Detector",
          "Dense(64, ReLU) -> BatchNorm -> Dropout(0.3)",
          "Dense(32) -> Dense(16) -> Dense(1, Sigmoid)"],
         "#6A1B9A", fontsize=8)
arrow(ax, 11.0, 7.6, 8.5, 6.85, color="#9C27B0")

# Row 3 — Metrics
for i, (lbl, val) in enumerate([("Accuracy", "~99.9%"), ("Precision", "~96%"),
                                   ("Recall", "~84%"), ("F1 Score", "~90%")]):
    draw_box(ax, 2.0 + i*3.7, 4.8, 3.2, 0.8, [lbl, val], "#2E7D32", fontsize=9)
arrow(ax, 8.5, 5.75, 8.5, 5.2, color="#4CAF50")

# Row 4 — Attacks
draw_box(ax, 3.0, 3.1, 4.0, 1.2,
         ["FGSM Attack", "x_adv = x + e*sign(grad)", "eps=0.1  |  1 step"],
         "#B71C1C")
draw_box(ax, 8.5, 3.1, 4.0, 1.2,
         ["PGD Attack", "Iterated FGSM + Projection", "eps=0.1  |  40 steps"],
         "#C62828")
draw_box(ax, 14.0, 3.1, 4.0, 1.2,
         ["Evasion Result", "Fraud -> Classified Normal", "Evasion rate measured"],
         "#880E4F")

arrow(ax, 8.5, 4.4, 3.0, 3.7, color="#FF5722")
arrow(ax, 8.5, 4.4, 8.5, 3.7, color="#FF5722")
arrow(ax, 10.5, 3.1, 12.0, 3.1, color="#FF7043")

# Row 5 — Defense
draw_box(ax, 3.5, 1.3, 5.0, 1.2,
         ["Adversarial Training", "Train on clean + adversarial", "FGSM ratio=0.5 per batch"],
         "#1B5E20")
draw_box(ax, 9.5, 1.3, 5.0, 1.2,
         ["Feature Squeezing", "Reduce input precision", "Removes tiny perturbations"],
         "#2E7D32")
draw_box(ax, 15.0, 1.3, 3.0, 1.2,
         ["Streamlit", "Dashboard", "5 interactive pages"],
         "#37474F")

arrow(ax, 3.0, 2.5, 3.5, 1.9, color="#66BB6A")
arrow(ax, 8.5, 2.5, 9.5, 1.9, color="#66BB6A")
arrow(ax, 12.0, 1.3, 13.5, 1.3, color="#90A4AE")

# Legend
patches = [
    mpatches.Patch(color="#1565C0", label="Data Pipeline"),
    mpatches.Patch(color="#6A1B9A", label="ML Model"),
    mpatches.Patch(color="#B71C1C", label="Attacks (FGSM/PGD)"),
    mpatches.Patch(color="#1B5E20", label="Defenses"),
    mpatches.Patch(color="#37474F", label="Dashboard"),
]
ax.legend(handles=patches, loc="lower center", ncol=5,
          facecolor="#1a1a3a", edgecolor="#555577",
          labelcolor="white", fontsize=9,
          bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig("diagrams/architecture.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Architecture diagram saved -> diagrams/architecture.png")
