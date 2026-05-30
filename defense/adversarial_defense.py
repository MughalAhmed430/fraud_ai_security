"""
defense/adversarial_defense.py
--------------------------------
Implements two defenses against adversarial attacks:

1. Adversarial Training
2. Feature Squeezing

🎓 Child-friendly explanation:
-------------------------------------------------------
Think of adversarial training like a fire drill:
  - We intentionally show the model "tricked" examples during training
  - The model learns: "Ahh, even when someone slightly changes a transaction,
    I should still recognise it as fraud!"
  - Like training a guard by having colleagues wear disguises to test them

Feature Squeezing is like wearing glasses that blur tiny details:
  - Small adversarial perturbations get "smoothed out"
  - The important features (large patterns) survive
  - Tiny noise that attackers added disappears
-------------------------------------------------------
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentNumpy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os


class OneHotCompatibleBCE(tf.keras.losses.Loss):
    """
    BinaryCrossEntropy that handles both binary (n,1) and one-hot (n,2) labels.
    ART's AdversarialTrainer always sends one-hot (n,2) to the loss, but our
    model has a single sigmoid output (n,1), which causes a shape mismatch.
    This wrapper converts one-hot to binary before computing the loss.
    """
    def call(self, y_true, y_pred):
        # ART sends one-hot (n,2); convert to binary (n,1) for sigmoid output
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.cond(
            tf.equal(tf.shape(y_true)[-1], 2),
            lambda: y_true[:, 1:2],   # take the fraud-class column
            lambda: y_true,
        )
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)


# ──────────────────────────────────────────────────────────────────────────────
# Defense 1: Adversarial Training
# ──────────────────────────────────────────────────────────────────────────────

def build_defended_model(input_dim: int) -> keras.Model:
    """Same architecture as the baseline — defense comes from training, not structure."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    return model


def adversarial_training(X_train, y_train, X_test, y_test,
                          eps: float = 0.05,
                          epochs: int = 10,
                          batch_size: int = 512,
                          save_path: str = "defense/defended_model.h5"):
    """
    Adversarial training: each epoch, we augment half of each mini-batch with
    FGSM adversarial examples and train the model on the mixed batch.

    Uses a direct Keras training loop (not ART's AdversarialTrainer) to avoid
    excessive tf.function retracing in ART 1.18+ with TF 2.18+.
    """
    print("\n🛡️  Starting Adversarial Training...")
    print(f"   Attack used for training : FGSM (ε={eps})")
    print(f"   Epochs                   : {epochs}")

    model = build_defended_model(X_train.shape[1])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # ART classifier — used only for generating FGSM perturbations
    classifier = TensorFlowV2Classifier(
        model=model,
        nb_classes=2,
        input_shape=(X_train.shape[1],),
        loss_object=OneHotCompatibleBCE(),
        clip_values=(-10.0, 10.0),
    )
    fgsm = FastGradientMethod(estimator=classifier, eps=eps)

    X_tr = X_train.astype(np.float32)
    y_tr = y_train.astype(np.float32)
    n = len(X_tr)

    for epoch in range(epochs):
        # Shuffle data each epoch
        idx = np.random.permutation(n)
        X_sh, y_sh = X_tr[idx], y_tr[idx]

        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        for start in range(0, n, batch_size):
            Xb = X_sh[start:start + batch_size]
            yb = y_sh[start:start + batch_size]

            # Replace 50% of the batch with FGSM adversarial examples
            half = len(Xb) // 2
            if half > 0:
                Xb_adv = fgsm.generate(x=Xb[:half])
                Xb = np.vstack([Xb_adv, Xb[half:]])
                # y labels stay the same — adversarial samples keep their true label

            hist = model.train_on_batch(Xb, yb[:len(Xb)], return_dict=True)
            epoch_loss += hist.get("loss", 0)
            epoch_acc  += hist.get("accuracy", 0)
            n_batches  += 1

        val_hist = model.evaluate(X_test.astype(np.float32), y_test.astype(np.float32),
                                   verbose=0, return_dict=True)
        print(f"   Epoch {epoch+1:2d}/{epochs} — "
              f"loss: {epoch_loss/n_batches:.4f}  acc: {epoch_acc/n_batches:.4f}  "
              f"val_loss: {val_hist['loss']:.4f}  val_acc: {val_hist['accuracy']:.4f}")

    os.makedirs("defense", exist_ok=True)
    model.save(save_path)
    print(f"\n💾 Defended model saved → {save_path}")

    return model, classifier


# ──────────────────────────────────────────────────────────────────────────────
# Defense 2: Feature Squeezing
# ──────────────────────────────────────────────────────────────────────────────

def feature_squeezing(X: np.ndarray, bit_depth: int = 4) -> np.ndarray:
    """
    Feature Squeezing: reduce the precision of each feature value.

    Why it works:
      Adversarial perturbations are tiny, precisely crafted nudges.
      Rounding values to fewer decimal places destroys those tiny nudges.
      The important fraud signals (large feature values) survive rounding.

    bit_depth=4 means we keep 4 significant bits of information per feature.
    Think of it like converting a high-res photo to a thumbnail —
    small scratches disappear, but faces are still recognisable.
    """
    # Normalise → quantise → restore
    X_min = X.min(axis=0)
    X_max = X.max(axis=0) + 1e-8   # avoid division by zero

    X_norm = (X - X_min) / (X_max - X_min)          # scale to [0, 1]
    levels = 2 ** bit_depth - 1                       # e.g. 15 for 4 bits
    X_squeezed = np.round(X_norm * levels) / levels   # quantise
    X_restored = X_squeezed * (X_max - X_min) + X_min # restore original scale

    return X_restored.astype(np.float32)


def evaluate_with_squeezing(model, X_original, X_adv, y_true,
                              attack_name: str = "Attack",
                              threshold: float = 0.5):

    # Squeeze the adversarial samples
    X_adv_squeezed = feature_squeezing(X_adv)

    prob_adv      = model.predict(X_adv,          verbose=0).flatten()
    prob_squeezed = model.predict(X_adv_squeezed, verbose=0).flatten()

    pred_adv      = (prob_adv      >= threshold).astype(int)
    pred_squeezed = (prob_squeezed >= threshold).astype(int)

    acc_adv      = accuracy_score(y_true, pred_adv)
    acc_squeezed = accuracy_score(y_true, pred_squeezed)

    rec_adv      = recall_score(y_true, pred_adv,      zero_division=0)
    rec_squeezed = recall_score(y_true, pred_squeezed, zero_division=0)

    print(f"\n{'─'*50}")
    print(f"  🧹 Feature Squeezing Defense — {attack_name}")
    print(f"{'─'*50}")
    print(f"  Detection (adversarial input) : {acc_adv*100:.1f}%  | Recall: {rec_adv*100:.1f}%")
    print(f"  Detection (after squeezing)   : {acc_squeezed*100:.1f}%  | Recall: {rec_squeezed*100:.1f}%")
    print(f"  Recovery improvement          : +{(acc_squeezed - acc_adv)*100:.1f}%")

    return {
        "acc_adv":       acc_adv,
        "acc_squeezed":  acc_squeezed,
        "rec_adv":       rec_adv,
        "rec_squeezed":  rec_squeezed,
        "X_squeezed":    X_adv_squeezed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full Comparison: Vulnerable vs Defended
# ──────────────────────────────────────────────────────────────────────────────

def compare_models(baseline_model, defended_model,
                   X_adv_fgsm, X_adv_pgd,
                   X_orig,     y_fraud,
                   threshold:  float = 0.5,
                   save_path: str = "dashboard/assets/defense_comparison.png"):
    """
    Side-by-side bar chart: Vulnerable vs Adversarially-trained model.
    """
    def get_recall(m, X, y):
        p = m.predict(X, verbose=0).flatten()
        pred = (p >= threshold).astype(int)
        return recall_score(y, pred, zero_division=0) * 100

    results = {
        "Clean Data":  [get_recall(baseline_model, X_orig,     y_fraud),
                        get_recall(defended_model,  X_orig,     y_fraud)],
        "After FGSM":  [get_recall(baseline_model, X_adv_fgsm, y_fraud),
                        get_recall(defended_model,  X_adv_fgsm, y_fraud)],
        "After PGD":   [get_recall(baseline_model, X_adv_pgd,  y_fraud),
                        get_recall(defended_model,  X_adv_pgd,  y_fraud)],
    }

    categories    = list(results.keys())
    baseline_vals = [v[0] for v in results.values()]
    defended_vals = [v[1] for v in results.values()]

    x     = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1a2e")

    b1 = ax.bar(x - width/2, baseline_vals, width,
                label="Vulnerable Model",  color="#FF4B4B", alpha=0.88)
    b2 = ax.bar(x + width/2, defended_vals, width,
                label="Defended Model (Adv. Training)", color="#69FF47", alpha=0.88)

    ax.set_ylabel("Fraud Detection Recall (%)", color="white")
    ax.set_title("Vulnerable vs Defended Model — Recall Under Attack",
                 color="white", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 120)
    ax.legend(fontsize=10, facecolor="#1a1a2e", labelcolor="white")
    ax.axhline(y=80, color="#aaa", linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, color="white")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    for bar in list(b1) + list(b2):
        ax.annotate(f"{bar.get_height():.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10,
                    color="white", fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"   Saved -> {save_path}")
    plt.close()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.prepare_data import load_and_prepare
    from attacks.adversarial_attacks import (
        wrap_model_for_art, run_fgsm, run_pgd
    )

    X_train, X_test, y_train, y_test, _, _ = load_and_prepare()

    baseline = keras.models.load_model("models/fraud_model.h5")
    art_cls  = wrap_model_for_art(baseline, input_shape=(X_test.shape[1],))

    # Generate adversarial examples with the baseline model
    X_adv_f, X_orig, y_fraud, _ = run_fgsm(art_cls, X_test, y_test, eps=0.1)
    X_adv_p, _,      _,       _ = run_pgd(art_cls,  X_test, y_test,
                                            eps=0.1, eps_step=0.01, max_iter=40)

    # ── Defense 1: Adversarial Training ───────────────────────────────────────
    defended_model, _ = adversarial_training(
        X_train, y_train, X_test, y_test,
        eps=0.05, epochs=10
    )

    # ── Defense 2: Feature Squeezing ──────────────────────────────────────────
    evaluate_with_squeezing(baseline, X_orig, X_adv_f, y_fraud, "FGSM")
    evaluate_with_squeezing(baseline, X_orig, X_adv_p, y_fraud, "PGD")

    # ── Comparison chart ──────────────────────────────────────────────────────
    compare_models(baseline, defended_model,
                   X_adv_f, X_adv_p, X_orig, y_fraud)
