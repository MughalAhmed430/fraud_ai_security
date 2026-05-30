"""
run_pipeline.py
---------------
Master script — runs the entire project end-to-end:

  Step 1 → Prepare data
  Step 2 → Train best-in-class fraud detection model
  Step 3 → Run FGSM + PGD attacks
  Step 4 → Apply defenses & compare results
  Step 5 → Save all artefacts for the Streamlit dashboard

Run from the project root:
    python run_pipeline.py
"""

import os
import sys
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, ".")

from data.prepare_data           import load_and_prepare
from models.train_model          import (compile_and_train, evaluate_model,
                                          find_best_threshold,
                                          plot_confusion_matrix,
                                          plot_training_history,
                                          plot_pr_curve)
from attacks.adversarial_attacks import (wrap_model_for_art, run_fgsm, run_pgd,
                                          measure_attack_effectiveness,
                                          plot_probability_shift,
                                          plot_feature_perturbation)
from defense.adversarial_defense import (adversarial_training,
                                          feature_squeezing,
                                          evaluate_with_squeezing,
                                          compare_models)
from tensorflow import keras

# ── Create output folders ─────────────────────────────────────────────────────
for d in ["models", "attacks", "defense", "dashboard/assets"]:
    os.makedirs(d, exist_ok=True)

print("\n" + "="*60)
print("  Financial AI Security — Fraud Detection Pipeline")
print("  Best-in-class model + Adversarial attacks + Defenses")
print("="*60)


# ═══════════════════════════════════════════════════════════════
# STEP 1 — Data
# ═══════════════════════════════════════════════════════════════
print("\n\nSTEP 1 — Loading & Preparing Data")
X_train, X_test, y_train, y_test, _, _ = load_and_prepare()


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Train Best Model
# ═══════════════════════════════════════════════════════════════
print("\n\nSTEP 2 — Training Best-in-Class Fraud Detection Model")
print("   Architecture : Deep Residual Network (128→64→32)")
print("   Loss         : Focal Loss (gamma=2, alpha=0.25)")
print("   Metric       : PR-AUC (best for imbalanced data)")
print("   Extras       : Class weights + Optimal threshold selection")

model, history = compile_and_train(
    X_train, y_train, X_test, y_test,
    epochs=30, batch_size=512,
    save_path="models/fraud_model.h5"
)

threshold, _ = find_best_threshold(model, X_test, y_test)
baseline_results = evaluate_model(model, X_test, y_test,
                                   title="Best Fraud Detector",
                                   threshold=threshold)
# Save optimal threshold alongside model
joblib.dump(threshold, "models/optimal_threshold.pkl")

plot_confusion_matrix(
    y_test, baseline_results["y_pred"],
    title="Best Model — Confusion Matrix",
    save_path="dashboard/assets/cm_baseline.png"
)
plot_training_history(history, save_path="dashboard/assets/training_history.png")
plot_pr_curve(y_test, baseline_results["y_prob"],
              save_path="dashboard/assets/pr_curve.png")

np.save("models/X_test.npy",  X_test)
np.save("models/y_test.npy",  y_test)


# ═══════════════════════════════════════════════════════════════
# STEP 3 — Adversarial Attacks
# ═══════════════════════════════════════════════════════════════
print("\n\nSTEP 3 — Running Adversarial Attacks")

classifier = wrap_model_for_art(model, input_shape=(X_test.shape[1],))

# FGSM
X_adv_f, X_orig_f, y_fraud_f, _ = run_fgsm(classifier, X_test, y_test, eps=0.1)
res_fgsm = measure_attack_effectiveness(
    model, X_orig_f, X_adv_f, y_fraud_f, "FGSM", threshold=threshold
)

# PGD
X_adv_p, X_orig_p, y_fraud_p, _ = run_pgd(
    classifier, X_test, y_test, eps=0.1, eps_step=0.01, max_iter=40
)
res_pgd = measure_attack_effectiveness(
    model, X_orig_p, X_adv_p, y_fraud_p, "PGD", threshold=threshold
)

plot_probability_shift(res_fgsm, res_pgd,
                        save_path="dashboard/assets/attack_comparison.png")
plot_feature_perturbation(X_orig_f, X_adv_f, X_adv_p,
                           save_path="dashboard/assets/feature_perturbation.png")

np.save("models/X_adv_fgsm.npy",   X_adv_f)
np.save("models/X_adv_pgd.npy",    X_adv_p)
np.save("models/X_orig_fraud.npy", X_orig_f)
np.save("models/y_fraud.npy",      y_fraud_f)


# ═══════════════════════════════════════════════════════════════
# STEP 4 — Defenses
# ═══════════════════════════════════════════════════════════════
print("\n\nSTEP 4 — Training Defended Model & Applying Defenses")

defended_model, _ = adversarial_training(
    X_train, y_train, X_test, y_test,
    eps=0.05, epochs=10,
    save_path="defense/defended_model.h5"
)

squeeze_fgsm = evaluate_with_squeezing(
    model, X_orig_f, X_adv_f, y_fraud_f, "FGSM", threshold=threshold
)
squeeze_pgd  = evaluate_with_squeezing(
    model, X_orig_f, X_adv_p, y_fraud_p, "PGD",  threshold=threshold
)

compare_models(
    model, defended_model,
    X_adv_f, X_adv_p, X_orig_f, y_fraud_f,
    threshold=threshold,
    save_path="dashboard/assets/defense_comparison.png"
)


# ═══════════════════════════════════════════════════════════════
# STEP 5 — Save summary for dashboard
# ═══════════════════════════════════════════════════════════════
print("\n\nSTEP 5 — Saving Pipeline Summary")

summary = {
    "baseline":     baseline_results,
    "fgsm":         res_fgsm,
    "pgd":          res_pgd,
    "squeeze_fgsm": {
        "acc_adv":      float(squeeze_fgsm["acc_adv"]),
        "acc_squeezed": float(squeeze_fgsm["acc_squeezed"]),
        "rec_adv":      float(squeeze_fgsm["rec_adv"]),
        "rec_squeezed": float(squeeze_fgsm["rec_squeezed"]),
    },
    "squeeze_pgd": {
        "acc_adv":      float(squeeze_pgd["acc_adv"]),
        "acc_squeezed": float(squeeze_pgd["acc_squeezed"]),
        "rec_adv":      float(squeeze_pgd["rec_adv"]),
        "rec_squeezed": float(squeeze_pgd["rec_squeezed"]),
    },
    "threshold": threshold,
}
for k in ["y_pred", "y_prob", "prob_before", "prob_after"]:
    summary["baseline"].pop(k, None)
    summary["fgsm"].pop(k, None)
    summary["pgd"].pop(k, None)

joblib.dump(summary, "models/pipeline_summary.pkl")
print("   Saved -> models/pipeline_summary.pkl")

print("\n\n" + "="*60)
print("  Pipeline Complete!")
print("="*60)
print("\n  Key Results:")
print(f"    Baseline Recall    : {baseline_results['recall']*100:.1f}%")
print(f"    Baseline PR-AUC    : {baseline_results['pr_auc']:.4f}")
print(f"    FGSM Evasion Rate  : {res_fgsm['evasion_rate']:.1f}%")
print(f"    PGD  Evasion Rate  : {res_pgd['evasion_rate']:.1f}%")
print(f"    Optimal Threshold  : {threshold:.4f}")
print("\n  Launch the dashboard:")
print("    streamlit run dashboard/app.py\n")
