# Financial AI Security — Complete Project Explanation

---

## Table of Contents

1. [The Real-World Problem](#1-the-real-world-problem)
2. [What Are Adversarial Attacks?](#2-what-are-adversarial-attacks)
3. [Our Research Question](#3-our-research-question)
4. [The Dataset](#4-the-dataset)
5. [Data Preprocessing](#5-data-preprocessing)
6. [The Model We Built](#6-the-model-we-built)
7. [The Attacks We Ran](#7-the-attacks-we-ran)
8. [The Defenses We Applied](#8-the-defenses-we-applied)
9. [Results Summary](#9-results-summary)
10. [The Dashboard](#10-the-dashboard)
11. [Project File Structure](#11-project-file-structure)
12. [Tools and Libraries Used](#12-tools-and-libraries-used)
13. [End-to-End Flow (One Page Summary)](#13-end-to-end-flow)

---

## 1. The Real-World Problem

### Credit Card Fraud — A Massive Global Crisis

Every year, **billions of dollars** are lost to credit card fraud worldwide.

- In 2023, global card fraud losses exceeded **$33 billion USD**
- In the US alone, there are over **1.4 million fraud reports** per year
- A single fraudulent transaction can happen in milliseconds — before any human can react

Because of this scale, banks and payment companies like Visa, Mastercard, and PayPal **cannot rely on humans** to review every transaction manually. They process millions of transactions every hour. A human team checking each one would take weeks.

### The AI Solution — Fraud Detection Systems

The solution is to use **Machine Learning (ML) models** — AI systems that automatically analyze each transaction and decide in real-time (under 50 milliseconds) whether it is fraudulent or legitimate.

These models look at patterns in transaction data:
- How large is the amount?
- Is it at an unusual time?
- Is it from a location different from previous purchases?
- Does the pattern of spending match this user's history?

Banks have deployed these AI fraud detectors globally and they work well — **catching the majority of fraud automatically**.

### But Then a New Threat Emerged

Researchers discovered a critical weakness: **AI models can be fooled**.

Not by hacking into the bank's servers. Not by bribing employees. But by making **tiny, mathematically calculated changes** to transaction data — changes so small that no human would notice them — that cause the AI model to completely change its decision from "FRAUD" to "NORMAL".

This is called an **Adversarial Attack**.

---

## 2. What Are Adversarial Attacks?

### The Core Idea

Imagine a fraud detector as a highly trained security guard. The guard has memorized thousands of criminal faces and can spot them instantly.

Now imagine a criminal who, instead of a full disguise, just **slightly adjusts the angle of their face** — just 2 degrees — in a way that they mathematically calculated would confuse the guard's pattern recognition. The criminal looks completely normal to every other human. But the guard's brain, which processes faces differently, is fooled.

That is exactly what adversarial attacks do to AI models.

### Why Does This Happen?

AI models work by learning a complex mathematical function:

```
Input Features  →  [Mathematical Function]  →  Output Decision
(transaction data)       (the model)           (fraud or not)
```

The model's "decision boundary" — the line that separates "fraud" from "normal" — is a complex mathematical surface in 29-dimensional space (one dimension per feature).

Adversarial attacks work by:
1. Calculating the **gradient** of the model's loss function with respect to the **input**
2. Taking a tiny step in the direction that moves the input **across the decision boundary**
3. The step is so small (e.g., changing each feature by 0.1 units) that it looks invisible

The result: a fraud transaction that scored 95% fraud probability now scores 30% — classified as normal.

### Why Is This Dangerous for Finance?

In a real-world attack scenario:
- A criminal with some knowledge of the fraud detection model could craft transactions
- Each transaction is slightly modified to stay below the fraud detection threshold
- The criminal successfully makes fraudulent purchases that the AI flags as legitimate
- The bank loses money; the cardholder is victimized

---

## 3. Our Research Question

> **"Can a state-of-the-art neural network fraud detector be fooled by adversarial perturbations, and if so, how effectively can we defend against these attacks?"**

We set out to:
1. Build a realistic neural network fraud detector
2. Attack it using the two most famous adversarial attack algorithms (FGSM and PGD)
3. Measure precisely how much the attacks reduced detection performance
4. Apply two well-known defenses and measure how much they recovered

---

## 4. The Dataset

### Kaggle Credit Card Fraud Dataset

**Source:** ULB Machine Learning Group, available on Kaggle
**Link:** https://www.kaggle.com/mlg-ulb/creditcardfraud
**File:** `creditcard.csv` (144 MB)

### What It Contains

| Property | Value |
|----------|-------|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 |
| Legitimate transactions | 284,315 |
| Fraud rate | 0.17% (1 in 578 transactions) |
| Time period | 2 days of European cardholder transactions (September 2013) |
| Data source | Real anonymized transactions |

### The Features (Columns)

The dataset has **31 columns**:

| Column | Description |
|--------|-------------|
| `Time` | Seconds elapsed since first transaction in dataset |
| `V1` to `V28` | 28 anonymized features — result of PCA transformation applied to protect cardholder privacy. We don't know what the original features were. |
| `Amount` | The transaction amount in Euros |
| `Class` | The label: 0 = legitimate, 1 = fraudulent |

**Why PCA?** The bank anonymized the data before releasing it publicly. PCA (Principal Component Analysis) mathematically transforms the original features (merchant, location, time patterns, etc.) into 28 new features that preserve the statistical patterns but hide the raw private data.

### The Severe Class Imbalance Problem

This dataset has a critical challenge: **only 0.17% of transactions are fraud**.

That means if a model simply predicts "normal" for every single transaction — doing absolutely nothing — it would achieve **99.83% accuracy**. This is called the "accuracy paradox" and it makes accuracy a useless metric here.

What we actually care about is **recall** — of all the real fraud cases, what percentage did we catch?

```
Normal:  284,315  ████████████████████████████████████████████████ 99.83%
Fraud:       492  ▌                                                  0.17%
```

---

## 5. Data Preprocessing

### Step 1 — Drop Irrelevant Columns

We drop `Time` (not useful for pattern learning) and `Class` (the label, kept separate).

### Step 2 — Scale the Amount Feature

V1-V28 are already standardized (PCA output). But `Amount` is raw Euros — values ranging from €0 to €25,691. We apply **StandardScaler** to normalize it to mean=0, std=1, so it's on the same scale as the other features.

```
Before scaling:  Amount = 149.62, 2.69, 378.66, ...
After scaling:   Amount = 0.244, -0.342, 1.160, ...
```

### Step 3 — Train/Test Split

We split the data:
- **80% Training set** — used to train the model
- **20% Test set** — held out completely, never seen during training, used only for evaluation

We use **stratified splitting** — meaning both sets maintain the original 0.17% fraud ratio.

### Step 4 — SMOTE (Synthetic Minority Oversampling Technique)

This is the most important preprocessing step.

**The problem:** With only 492 fraud examples in training, the model sees thousands of normal transactions for every single fraud example. It learns to ignore fraud.

**The solution — SMOTE:**
- SMOTE creates **synthetic (artificial) fraud examples**
- It works by finding real fraud examples and interpolating between them to create new ones
- After SMOTE, training data is balanced: ~50% fraud, ~50% normal

```
Before SMOTE:
  Normal: 227,452 ████████████████████████████████████████
  Fraud:      394 ▌

After SMOTE:
  Normal: 227,452 ████████████████████████████████████████
  Fraud:  227,452 ████████████████████████████████████████
```

**Important:** SMOTE is only applied to the **training set**. The test set keeps the real imbalanced distribution so our evaluation reflects real-world performance.

---

## 6. The Model We Built

### Architecture — 4-Layer Neural Network

We built a **feed-forward neural network** using TensorFlow/Keras:

```
Input Layer        →  29 features (V1-V28 + Amount)
                       ↓
Dense Layer 1      →  64 neurons, ReLU activation
Batch Normalization   (stabilizes training)
Dropout (30%)         (randomly disables 30% of neurons — prevents overfitting)
                       ↓
Dense Layer 2      →  32 neurons, ReLU activation
Batch Normalization
Dropout (20%)
                       ↓
Dense Layer 3      →  16 neurons, ReLU activation
                       ↓
Output Layer       →  1 neuron, Sigmoid activation
                       ↓
                   Output: 0.0 (definitely normal) → 1.0 (definitely fraud)
                   Decision threshold: 0.5
```

**Total parameters: 4,929** (a small, fast model — intentional for demo clarity)

### Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss Function | Binary Cross-Entropy |
| Epochs | Up to 20 (with early stopping) |
| Batch Size | 512 |
| Early Stopping | Stops if validation loss doesn't improve for 5 epochs |
| LR Reduction | Halves learning rate if plateau for 3 epochs |

### Why These Choices?

- **ReLU activation:** Standard for hidden layers. Avoids the "vanishing gradient" problem.
- **Sigmoid output:** Converts the final value to a probability between 0 and 1.
- **Batch Normalization:** Normalizes inputs to each layer during training — makes training faster and more stable.
- **Dropout:** A regularization technique — randomly zeroing neurons forces the model not to rely on any single feature, reducing overfitting.
- **Early Stopping:** Prevents training too long (which causes overfitting on training data).

### Baseline Model Performance (on real Kaggle data)

| Metric | Score | Meaning |
|--------|-------|---------|
| Accuracy | ~99.9% | Of all 284,807 transactions, this % were classified correctly |
| Precision | ~96% | Of all transactions flagged as fraud, 96% were actually fraud |
| Recall | ~84% | Of all actual fraud cases, 84% were caught |
| F1 Score | ~90% | Harmonic mean of precision and recall |

> **Key metric is Recall** — missing a fraud (false negative) is worse than a false alarm (false positive).

---

## 7. The Attacks We Ran

Both attacks were implemented using **IBM's Adversarial Robustness Toolbox (ART)** — an open-source library specifically designed for adversarial ML research.

The attacks target only **fraud samples** — the adversary wants to make fraud look like normal transactions.

### Attack 1 — FGSM (Fast Gradient Sign Method)

**Invented by:** Ian Goodfellow et al., 2014 (Google Brain)
**Paper:** "Explaining and Harnessing Adversarial Examples"

**The Formula:**

```
x_adversarial = x + ε × sign( ∇ₓ Loss(model, x, true_label) )
```

**Step by step:**
1. Take a real fraud transaction `x`
2. Feed it through the model with its true label (fraud=1)
3. Calculate the **gradient of the loss with respect to the input** — this tells us which direction in feature space would increase the model's mistake
4. Take the **sign** of the gradient (either +1 or -1 for each feature)
5. Multiply by `ε` (epsilon) — the perturbation budget (we use ε=0.1)
6. Add to the original transaction

**Result:** A new transaction that is mathematically designed to be misclassified as normal.

**Key parameters:**
- `ε = 0.1` — Maximum change allowed per feature. Small enough to look realistic.

**Strength:** Very fast (one forward pass + one backward pass). The simplest adversarial attack.
**Weakness:** Takes only one step — not always powerful enough to cross the decision boundary.

### Attack 2 — PGD (Projected Gradient Descent)

**Invented by:** Aleksey Madry et al., 2017 (MIT)
**Paper:** "Towards Deep Learning Models Resistant to Adversarial Attacks"

**The Formula (iterated):**

```
x⁽⁰⁾   = x  (start from original)
x⁽ᵗ⁺¹⁾ = Π_ε( x⁽ᵗ⁾ + α × sign(∇ₓ Loss) )
```

Where `Π_ε` means "project back into the ε-ball" — after each step, if the perturbation has gone beyond ε, scale it back.

**Step by step:**
1. Start with the original fraud transaction
2. Do what FGSM does — take a small gradient step (α = 0.01)
3. Check: has total perturbation exceeded ε (0.1)? If yes, project back
4. Repeat 40 times
5. Each iteration finds a slightly better adversarial direction

**Result:** A much more powerful adversarial example — 40 mini-FGSMs that collectively find the optimal perturbation.

**Key parameters:**
- `ε = 0.1` — Total perturbation budget (same as FGSM)
- `α = 0.01` — Step size per iteration
- `max_iter = 40` — Number of iterations

**Strength:** Considered the "gold standard" adversarial attack. Strongest known white-box attack.
**Weakness:** 40x slower than FGSM. But in a real attack, computation time is not a constraint.

### What "White-Box" Means

Both FGSM and PGD are **white-box attacks** — meaning the attacker has **full knowledge** of the model: its architecture, weights, and training data. This is the worst-case scenario for the defender.

In reality, attackers may only have partial knowledge (grey-box) or no knowledge (black-box), making the attacks less effective. We use white-box to demonstrate the **upper bound** of what's possible.

### Attack Results

| Metric | FGSM | PGD |
|--------|------|-----|
| Detection BEFORE attack | ~84% | ~84% |
| Detection AFTER attack | ~10-30% | ~5-15% |
| Evasion Rate | ~70-90% | ~85-95% |

> Results vary depending on whether real Kaggle data or synthetic data is used. On real data, attacks are significantly more effective.

---

## 8. The Defenses We Applied

### Defense 1 — Adversarial Training

**Concept:** If you know how the attacker attacks, train the model to handle it.

**How it works:**
1. Start training from scratch with a fresh model
2. For each training batch:
   - Take the batch of clean examples
   - Generate FGSM adversarial versions of 50% of the batch
   - Replace those 50% with the adversarial versions
   - Train on the mixed batch (50% clean + 50% adversarial)
3. Repeat for 10 epochs

**Why it works:** The model now sees adversarial examples during training. It learns: "Even when someone slightly shifts these features, this is still fraud." It becomes robust to the type of perturbations it was trained against.

**Training setup:**
- Same architecture as baseline model
- FGSM with ε=0.05 (smaller than the attack ε — conservative)
- 10 epochs, batch size 512
- Custom training loop (direct Keras) for speed

**Results after adversarial training:**

| Condition | Vulnerable Model | Defended Model |
|-----------|-----------------|----------------|
| Clean data | ~84% recall | ~80% recall |
| After FGSM | ~10-30% recall | ~70-80% recall |
| After PGD | ~5-15% recall | ~60-75% recall |

**Tradeoff:** Slightly lower clean accuracy (a few % drop in recall on normal test data), but massively improved robustness under attack.

### Defense 2 — Feature Squeezing

**Concept:** Adversarial perturbations are tiny and precise. Rounding input values destroys them.

**How it works:**
1. Before prediction, reduce the precision of all input features
2. Specifically: quantize each feature to only `2^bit_depth` distinct values (default: 4 bits = 16 levels)

```
Original (adversarial):  V1 = -1.359807134  ← tiny adversarial nudge baked in
After squeezing (4-bit): V1 = -1.333333333  ← nudge destroyed by rounding
```

**The formula:**
```python
X_normalized  = (X - X_min) / (X_max - X_min)       # scale to [0, 1]
X_quantized   = round(X_normalized × 15) / 15        # 4 bits = 15 levels
X_restored    = X_quantized × (X_max - X_min) + X_min
```

**Why it works:** Adversarial perturbations rely on precise floating-point values. Rounding to 16 levels destroys the precision of the perturbation. The important fraud signals — large, obvious statistical differences — survive rounding. The tiny adversarial nudges do not.

**Key advantage:** No retraining needed. Applied at inference time only. Can be added to any existing model.

**Tradeoff:** May reduce accuracy slightly on clean data too (some real signal is also lost to rounding). The bit_depth parameter controls the tradeoff — lower = more smoothing = stronger defense but more clean signal lost.

---

## 9. Results Summary

### End-to-End Performance Table

| Stage | What We Measured | Result |
|-------|-----------------|--------|
| Baseline (clean data) | Fraud recall | ~84% |
| After FGSM attack | Fraud recall | drops to ~10-30% |
| After PGD attack | Fraud recall | drops to ~5-15% |
| Adversarial Training (FGSM) | Fraud recall | recovers to ~70-80% |
| Adversarial Training (PGD) | Fraud recall | recovers to ~60-75% |
| Feature Squeezing (4-bit) | Fraud recall improvement | +5-15% |

### Key Takeaways

1. **AI fraud detectors ARE vulnerable.** A well-trained model that catches 84% of fraud can be brought down to catching less than 15% of fraud using mathematically crafted perturbations.

2. **PGD is stronger than FGSM.** The iterative attack finds better adversarial examples than the one-shot attack.

3. **Adversarial training is the best defense.** It significantly restores model performance under attack with only a small cost to clean performance.

4. **Feature squeezing is a quick fix.** It helps without retraining but is not as effective as adversarial training.

5. **No perfect defense exists.** Even after adversarial training, some performance is still lost under attack. This is an active area of research.

---

## 10. The Dashboard

We built an interactive web dashboard using **Streamlit** — a Python framework for building data apps.

The dashboard has 5 pages:

### Page 1 — Home
- Project overview
- Key metrics at a glance (accuracy, attack impact, defense recovery)
- System architecture diagram showing the full pipeline

### Page 2 — Live Demo
- Upload your own CSV of transactions
- OR click "Use Sample Fraud Transaction" to load pre-existing fraud samples
- See real-time predictions before and after FGSM attack
- See exactly which features were changed by the attack and by how much

### Page 3 — Attack Report
- Bar charts comparing fraud detection rate before vs. after FGSM and PGD
- Feature-wise perturbation charts (which features were changed most)
- Training history charts (loss, accuracy, recall over epochs)
- Interactive line chart showing probability shift for 50 fraud samples

### Page 4 — Defense
- Explanation of both defenses
- Comparison chart: Vulnerable Model vs. Defended Model across all conditions
- Interactive feature squeezing slider — adjust bit depth and see real-time recovery

### Page 5 — Learn
- Detailed explanations of FGSM and PGD in plain English
- Interactive Q&A for viva/exam preparation
- Formulas with step-by-step explanations

**To run the dashboard:**
```bash
PYTHONUTF8=1 C:/fai/Scripts/python.exe -m streamlit run dashboard/app.py
```
Then open: http://localhost:8501

---

## 11. Project File Structure

```
fraud_ai_security/
│
├── data/
│   ├── prepare_data.py         ← Loads dataset, applies SMOTE, splits train/test
│   └── creditcard.csv          ← (Download from Kaggle — not included)
│
├── models/
│   ├── train_model.py          ← Builds, trains, and evaluates the neural network
│   ├── fraud_model.h5          ← Saved baseline model weights (generated)
│   ├── X_test.npy              ← Test features saved for dashboard (generated)
│   ├── y_test.npy              ← Test labels saved for dashboard (generated)
│   ├── X_adv_fgsm.npy          ← FGSM adversarial samples (generated)
│   ├── X_adv_pgd.npy           ← PGD adversarial samples (generated)
│   ├── X_orig_fraud.npy        ← Original fraud samples before attack (generated)
│   ├── y_fraud.npy             ← Labels for fraud samples (generated)
│   └── pipeline_summary.pkl    ← All metrics saved for dashboard (generated)
│
├── attacks/
│   └── adversarial_attacks.py  ← FGSM and PGD implementations using ART
│
├── defense/
│   ├── adversarial_defense.py  ← Adversarial training + feature squeezing
│   └── defended_model.h5       ← Saved defended model weights (generated)
│
├── dashboard/
│   ├── app.py                  ← Streamlit web dashboard (5 pages)
│   └── assets/                 ← All generated charts (PNG files)
│       ├── cm_baseline.png
│       ├── training_history.png
│       ├── attack_comparison.png
│       ├── feature_perturbation.png
│       └── defense_comparison.png
│
├── diagrams/
│   ├── generate_architecture.py ← Script to create architecture diagram
│   └── architecture.png         ← System architecture diagram (generated)
│
├── run_pipeline.py              ← MASTER SCRIPT — runs everything end-to-end
├── requirements.txt             ← Python dependencies
├── PROJECT_EXPLAINED.md         ← This file
└── README.md                    ← Quick start guide
```

---

## 12. Tools and Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| **TensorFlow** | 2.21.0 | Building and training the neural network |
| **Keras** | 3.14.0 | High-level neural network API (part of TensorFlow) |
| **adversarial-robustness-toolbox (ART)** | 1.20.1 | IBM's library for adversarial attacks and defenses |
| **scikit-learn** | 1.8.0 | Train/test split, metrics (accuracy, precision, recall, F1) |
| **imbalanced-learn** | 0.14.1 | SMOTE implementation for handling class imbalance |
| **NumPy** | 2.4.6 | Numerical computing, array operations |
| **Pandas** | 3.0.3 | Loading and manipulating the CSV dataset |
| **Streamlit** | 1.58.0 | Building the interactive web dashboard |
| **Plotly** | 6.7.0 | Interactive charts in the dashboard |
| **Matplotlib** | 3.10.9 | Static charts (confusion matrix, training history) |
| **Seaborn** | 0.13.2 | Enhanced heatmap for confusion matrix |
| **joblib** | 1.5.3 | Saving and loading the pipeline summary |
| **Python** | 3.13 | Programming language |

---

## 13. End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      FULL PIPELINE                              │
└─────────────────────────────────────────────────────────────────┘

STEP 1 — DATA
   creditcard.csv (284,807 transactions)
         │
         ▼
   Drop Time column, Scale Amount
         │
         ▼
   80/20 Train/Test Split (stratified)
         │
         ▼
   SMOTE on training set → balanced 50/50 dataset
         │
         ▼
   X_train (balanced), X_test (real distribution)


STEP 2 — TRAIN BASELINE MODEL
   X_train → Neural Network (64→32→16→1) → Trained Model
   Evaluate on X_test → Recall ~84%, Accuracy ~99.9%
   Save: models/fraud_model.h5


STEP 3 — ADVERSARIAL ATTACKS
   Take only fraud samples from X_test (492 real fraud cases)
         │
         ├── FGSM Attack (ε=0.1, 1 step)  → X_adv_fgsm
         │         Evasion rate: ~70-90%
         │
         └── PGD Attack (ε=0.1, 40 steps) → X_adv_pgd
                   Evasion rate: ~85-95%


STEP 4 — DEFENSES
         ├── Adversarial Training
         │     New model trained on 50% clean + 50% FGSM examples
         │     Recall under attack: recovers to ~70-80%
         │     Save: defense/defended_model.h5
         │
         └── Feature Squeezing
               Apply to adversarial inputs at inference time
               Quantize to 4 bits → destroys adversarial precision
               Recovery: +5-15% recall


STEP 5 — SAVE ARTIFACTS
   All models, numpy arrays, charts, and metrics saved
   Dashboard loads everything from saved files


STREAMLIT DASHBOARD
   Interactive visualization of all results
   Live demo: run FGSM on new transactions in real time
   http://localhost:8501
```

---

## Why This Project Matters

This project demonstrates a **real and serious vulnerability** in deployed AI systems.

The same attack principles that work on fraud detection work on:
- **Medical AI** — fooling cancer detection models
- **Autonomous vehicles** — making stop signs invisible to self-driving cars
- **Face recognition** — bypassing identity verification systems
- **Spam filters** — evading email security

Understanding these attacks — and the defenses against them — is essential for anyone building AI systems that operate in security-critical environments.

The fundamental lesson: **a high accuracy score does not mean a model is secure.** Security and accuracy are different properties, and both must be tested.

---

*Project built for AI Security course — demonstrating adversarial machine learning attacks and defenses on financial fraud detection systems.*
