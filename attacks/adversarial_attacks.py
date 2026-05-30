"""
attacks/adversarial_attacks.py
-------------------------------
Implements FGSM and PGD adversarial attacks using IBM's ART library.

🎓 Child-friendly explanation:
-------------------------------------------------------
Imagine a fraud detector is like a teacher who marks tests.

FGSM (Fast Gradient Sign Method):
  "Take ONE tiny step in the worst possible direction"
  - Like a student who erases one answer just enough to confuse the teacher
  - Fast, one-shot, but not always effective

PGD (Projected Gradient Descent):
  "Take MANY tiny steps, each time staying within a small boundary"
  - Like a student who makes 40 tiny changes, each checked to not look too suspicious
  - Slower, but MUCH more powerful — the 'king' of adversarial attacks
-------------------------------------------------------
"""

import numpy as np
import tensorflow as tf
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentNumpy
import matplotlib.pyplot as plt
import os


# ──────────────────────────────────────────────────────────────────────────────
# Wrap Keras model for ART
# ──────────────────────────────────────────────────────────────────────────────

def wrap_model_for_art(model, input_shape, clip_values=(-10.0, 10.0)):
    """
    ART needs a wrapped classifier.
    
    clip_values: min/max feature values — keeps adversarial samples realistic.
    (Like telling the attacker: "you can't change a transaction amount by $1 million,
     that would be too obvious!")
    """
    classifier = TensorFlowV2Classifier(
        model=model,
        nb_classes=2,
        input_shape=input_shape,
        loss_object=tf.keras.losses.BinaryCrossentropy(),
        clip_values=clip_values,
    )
    return classifier


# ──────────────────────────────────────────────────────────────────────────────
# FGSM Attack
# ──────────────────────────────────────────────────────────────────────────────

def run_fgsm(classifier, X_test, y_test, eps: float = 0.1):
    """
    FGSM: Fast Gradient Sign Method (Goodfellow et al., 2014)
    
    Formula:  x_adv = x + ε * sign(∇_x L(θ, x, y))
    
    Plain English:
      1. Compute the gradient of the loss w.r.t. the INPUT features
         (ask: "which direction in feature space increases the model's mistake?")
      2. Take one step of size ε in that direction
    
    eps: perturbation size (like the volume knob — higher = more distortion)
    """
    print(f"\n⚡ Running FGSM Attack (ε={eps})...")
    attack = FastGradientMethod(estimator=classifier, eps=eps)

    # Only attack fraud samples (class=1) — the adversary wants fraud to look normal
    fraud_idx = np.where(y_test == 1)[0][:200]          # use first 200 fraud samples
    X_fraud   = X_test[fraud_idx].astype(np.float32)
    y_fraud   = y_test[fraud_idx]

    X_adv = attack.generate(x=X_fraud)
    perturbation = np.mean(np.abs(X_adv - X_fraud))
    print(f"   Mean perturbation magnitude: {perturbation:.6f}")

    return X_adv, X_fraud, y_fraud, fraud_idx


# ──────────────────────────────────────────────────────────────────────────────
# PGD Attack
# ──────────────────────────────────────────────────────────────────────────────

def run_pgd(classifier, X_test, y_test, eps: float = 0.1,
            eps_step: float = 0.01, max_iter: int = 40):
    """
    PGD: Projected Gradient Descent (Madry et al., 2017)
    
    Formula (iterated):
      x^(t+1) = Π_{x+S}( x^(t) + α * sign(∇_x L) )
    
    Plain English:
      1. Start with the original sample
      2. Take a small gradient step (like FGSM but tiny)
      3. "Project" back into the allowed budget ε (don't stray too far)
      4. Repeat max_iter times
    
    This is Madry's attack — considered the gold standard adversarial attack.
    """
    print(f"\n🔄 Running PGD Attack (ε={eps}, steps={max_iter})...")
    attack = ProjectedGradientDescentNumpy(
        estimator=classifier,
        eps=eps,
        eps_step=eps_step,
        max_iter=max_iter,
        targeted=False,
        verbose=False,
    )

    fraud_idx = np.where(y_test == 1)[0][:200]
    X_fraud   = X_test[fraud_idx].astype(np.float32)
    y_fraud   = y_test[fraud_idx]

    X_adv = attack.generate(x=X_fraud)
    perturbation = np.mean(np.abs(X_adv - X_fraud))
    print(f"   Mean perturbation magnitude: {perturbation:.6f}")

    return X_adv, X_fraud, y_fraud, fraud_idx


# ──────────────────────────────────────────────────────────────────────────────
# Measure Attack Effectiveness
# ──────────────────────────────────────────────────────────────────────────────

def measure_attack_effectiveness(model, X_original, X_adv, y_true,
                                  attack_name: str = "Attack",
                                  threshold: float = 0.5):
    """
    Compare model predictions before and after the attack.
    Uses the optimal threshold found during training.
    """
    prob_before = model.predict(X_original, verbose=0).flatten()
    prob_after  = model.predict(X_adv,      verbose=0).flatten()

    pred_before = (prob_before >= threshold).astype(int)
    pred_after  = (prob_after  >= threshold).astype(int)

    # For fraud samples: success = predicted as Normal (0) after attack
    total      = len(y_true)
    fooled     = np.sum((pred_before == 1) & (pred_after == 0))
    evasion_rate = fooled / max(np.sum(pred_before == 1), 1) * 100

    acc_before = np.mean(pred_before == y_true) * 100
    acc_after  = np.mean(pred_after  == y_true) * 100

    print(f"\n{'─'*50}")
    print(f"  📉 {attack_name} Results")
    print(f"{'─'*50}")
    print(f"  Detection rate BEFORE attack : {acc_before:.1f}%")
    print(f"  Detection rate AFTER  attack : {acc_after:.1f}%")
    print(f"  Transactions fooled          : {fooled}/{total}")
    print(f"  Evasion rate                 : {evasion_rate:.1f}%")

    return {
        "attack_name":   attack_name,
        "acc_before":    acc_before,
        "acc_after":     acc_after,
        "fooled":        fooled,
        "total":         total,
        "evasion_rate":  evasion_rate,
        "prob_before":   prob_before,
        "prob_after":    prob_after,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualisations
# ──────────────────────────────────────────────────────────────────────────────

def plot_probability_shift(results_fgsm, results_pgd, save_path=None):
    attacks = ["FGSM", "PGD"]
    before  = [results_fgsm["acc_before"], results_pgd["acc_before"]]
    after   = [results_fgsm["acc_after"],  results_pgd["acc_after"]]

    x = np.arange(len(attacks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1a2e")

    bars1 = ax.bar(x - width/2, before, width, label="Before Attack", color="#00E5FF", alpha=0.9)
    bars2 = ax.bar(x + width/2, after,  width, label="After Attack",  color="#FF4B4B", alpha=0.9)

    ax.set_ylabel("Fraud Detection Rate (%)", color="white")
    ax.set_title("Fraud Detection: Before vs After Attack", color="white",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=12, color="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.set_ylim(0, 115)
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    for bar in list(bars1) + list(bars2):
        ax.annotate(f'{bar.get_height():.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11,
                    color="white", fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"   Saved -> {save_path}")
    plt.close()
    return fig


def plot_feature_perturbation(X_original, X_adv_fgsm, X_adv_pgd,
                               n_features=15, save_path=None):
    fgsm_diff = np.mean(np.abs(X_adv_fgsm - X_original), axis=0)[:n_features]
    pgd_diff  = np.mean(np.abs(X_adv_pgd  - X_original), axis=0)[:n_features]

    feature_names = [f"V{i+1}" for i in range(n_features)]
    x = np.arange(n_features)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1a2e")

    ax.bar(x - width/2, fgsm_diff, width, label="FGSM", color="#FF9800", alpha=0.85)
    ax.bar(x + width/2, pgd_diff,  width, label="PGD",  color="#9C27B0", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, color="white")
    ax.set_ylabel("Mean Absolute Perturbation", color="white")
    ax.set_title("Feature-wise Perturbation Magnitude: FGSM vs PGD",
                 color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, joblib
    sys.path.insert(0, ".")
    from data.prepare_data  import load_and_prepare
    from tensorflow import keras

    os.makedirs("attacks", exist_ok=True)

    _, X_test, _, y_test, _, _ = load_and_prepare()

    model = keras.models.load_model("models/fraud_model.h5")
    classifier = wrap_model_for_art(model, input_shape=(X_test.shape[1],))

    # ── FGSM ──────────────────────────────────────────────────────────────────
    X_adv_f, X_orig_f, y_f, _ = run_fgsm(classifier, X_test, y_test, eps=0.1)
    res_fgsm = measure_attack_effectiveness(model, X_orig_f, X_adv_f, y_f, "FGSM")

    # ── PGD ───────────────────────────────────────────────────────────────────
    X_adv_p, X_orig_p, y_p, _ = run_pgd(classifier, X_test, y_test,
                                          eps=0.1, eps_step=0.01, max_iter=40)
    res_pgd = measure_attack_effectiveness(model, X_orig_p, X_adv_p, y_p, "PGD")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_probability_shift(res_fgsm, res_pgd, save_path="attacks/attack_comparison.png")
    plot_feature_perturbation(X_orig_f, X_adv_f, X_adv_p,
                               save_path="attacks/feature_perturbation.png")
