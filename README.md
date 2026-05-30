# 🔐 Financial AI Security — Evasion Attacks on Fraud Detection Systems

> **University Project** | AI Security & Cryptography Course  
> Demonstrates adversarial attacks (FGSM & PGD) on a neural network fraud detector,
> with practical defenses and an interactive Streamlit dashboard.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Folder Structure](#-folder-structure)
3. [Setup Guide](#-setup-guide)
4. [Running the Project](#-running-the-project)
5. [FGSM & PGD Explained](#-fgsm--pgd-explained)
6. [Architecture Diagram](#-architecture-diagram)
7. [Presentation Talking Points](#-presentation-talking-points)
8. [Viva Q&A](#-viva-qa)

---

## 🎯 Project Overview

| Component | Technology |
|-----------|-----------|
| Dataset   | Kaggle Credit Card Fraud (284,807 transactions) |
| Model     | TensorFlow/Keras Neural Network |
| Attacks   | FGSM & PGD via IBM Adversarial Robustness Toolbox |
| Defense   | Adversarial Training + Feature Squeezing |
| Dashboard | Streamlit interactive web app |

### What this project proves:
- A well-trained fraud detector can be **fooled by imperceptibly small changes**
- **FGSM** (one-shot attack) can reduce fraud detection from ~84% to near 0%
- **PGD** (iterative attack) is even more powerful
- **Adversarial training** significantly restores robustness
- **Feature squeezing** can neutralise small perturbations without retraining

---

## 📁 Folder Structure

```
fraud_ai_security/
│
├── data/
│   ├── prepare_data.py        ← load + preprocess Kaggle dataset
│   └── creditcard.csv         ← ⚠️ YOU MUST DOWNLOAD THIS (see step 3)
│
├── models/
│   ├── train_model.py         ← build, train, evaluate the neural network
│   ├── fraud_model.h5         ← saved baseline model (after training)
│   ├── X_test.npy             ← test features (saved for dashboard)
│   └── y_test.npy             ← test labels
│
├── attacks/
│   └── adversarial_attacks.py ← FGSM + PGD attack implementations
│
├── defense/
│   ├── adversarial_defense.py ← adversarial training + feature squeezing
│   └── defended_model.h5      ← saved defended model (after training)
│
├── dashboard/
│   ├── app.py                 ← Streamlit dashboard (5 pages)
│   └── assets/                ← auto-generated charts/images
│
├── diagrams/
│   ├── generate_architecture.py
│   └── architecture.png       ← auto-generated
│
├── run_pipeline.py            ← ⭐ MASTER SCRIPT — runs everything
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup Guide

### Step 1 — Clone / Download the project
```bash
# If using git:
git clone <your-repo-url>
cd fraud_ai_security

# Or just unzip the project folder and open a terminal inside it
```

### Step 2 — Create a Python virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
> ⏱️ This may take 3–5 minutes (TensorFlow + ART are large packages)

### Step 4 — Download the Dataset
1. Go to: https://www.kaggle.com/mlg-ulb/creditcardfraud
2. Download **creditcard.csv** (144 MB)
3. Place it in the `data/` folder:
   ```
   fraud_ai_security/data/creditcard.csv
   ```
> You need a free Kaggle account to download it.

### Step 5 — Generate the architecture diagram
```bash
python diagrams/generate_architecture.py
```

---

## 🚀 Running the Project

### Option A — Run Everything At Once (Recommended)
```bash
python run_pipeline.py
```
This will:
- ✅ Load and preprocess the data
- ✅ Train the baseline fraud detection model
- ✅ Run FGSM and PGD attacks
- ✅ Train the defended model
- ✅ Apply feature squeezing defense
- ✅ Save all charts and metrics
- ✅ Print a full results summary

> ⏱️ Total time: ~10–20 minutes (depending on your hardware)

### Option B — Run individual modules
```bash
# Train model only
python models/train_model.py

# Run attacks only (requires trained model)
python attacks/adversarial_attacks.py

# Run defenses only (requires trained model + adversarial samples)
python defense/adversarial_defense.py
```

### Option C — Launch the Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at: **http://localhost:8501**

> ⚠️ Run the pipeline first so the dashboard has data to show.

---

## 📐 FGSM & PGD Explained

### ⚡ FGSM — Fast Gradient Sign Method

**Formula:**
```
x_adversarial = x + ε × sign( ∇ₓ Loss(θ, x, y) )
```

**Step by step:**
1. Feed the original transaction `x` into the model
2. Calculate which direction in feature space increases the model's error (gradient)
3. Take ONE step of size ε in that direction
4. The result looks identical but fools the model

**Analogy:** A student who changes exactly one word on their exam to confuse the teacher's plagiarism detector.

**ε (epsilon):** Controls how big the perturbation is
- Small (0.01): subtle, hard to fool
- Large (0.3): obvious, easy to fool

---

### 🔄 PGD — Projected Gradient Descent

**Formula (repeated T times):**
```
x⁽⁰⁾ = x + Uniform(-ε, ε)           # random start inside budget
x⁽ᵗ⁺¹⁾ = Π_{B(x,ε)}( x⁽ᵗ⁾ + α × sign(∇ₓ Loss) )
```

**Step by step:**
1. Start from the original sample (with small random noise)
2. Take a tiny FGSM step
3. Project back into the "budget ball" (don't exceed ε)
4. Repeat 40 times
5. Keep the worst-case result

**Analogy:** A spy who makes 40 tiny changes to a fake passport, double-checking each change stays within "believable limits."

**Why stronger?** Finds the worst-case perturbation within the budget, not just any perturbation.

---

## 🏗️ Architecture Diagram

See `diagrams/architecture.png` (auto-generated by `diagrams/generate_architecture.py`)

```
Data → Preprocessing → Model → Evaluation
                          ↓
               Adversarial Attacks (FGSM / PGD)
                          ↓
                     Defenses Applied
                          ↓
                  Streamlit Dashboard
```

---

## 🎤 Presentation Talking Points

### Opening (30 seconds)
> "AI fraud detectors protect millions of bank transactions daily. But what happens 
> when an attacker knows the AI is watching? Today I'll show how small, invisible 
> changes to transaction data can completely fool a state-of-the-art neural network — 
> and how we can defend against it."

### The Model (1 minute)
- Trained on 284,807 real credit card transactions
- 99.9% accuracy on clean data — sounds great!
- But accuracy is misleading with only 0.17% fraud cases
- Recall matters more: "of all real fraud, how much do we catch?"

### The Attack (2 minutes)
- "Imagine a criminal who knows the bank's AI uses gradients to decide fraud"
- FGSM: one tiny nudge per feature = model completely confused
- PGD: 40 tiny nudges = even more powerful
- Show the demo: fraud probability drops from 90% → 5% after attack

### The Defense (1 minute)
- Adversarial training: "teach the bouncer by sending in undercover agents"
- Feature squeezing: "round off the tiny precision attackers rely on"
- Result: defended model maintains ~80% recall even under attack

### Conclusion (30 seconds)
> "This demonstrates a fundamental tension in AI security: the same gradient 
> information that makes neural networks trainable also makes them attackable.
> Real-world defenses must be layered — no single defense is perfect."

---

## 🎓 Viva Q&A

**Q1: What is an adversarial attack?**  
A carefully crafted perturbation added to a model's input that causes misclassification, while appearing normal to humans. In this project, we make fraudulent transactions appear legitimate.

**Q2: What's the difference between FGSM and PGD?**  
FGSM takes one large gradient step. PGD takes many small steps and "projects" back to stay within budget ε after each step. PGD is stronger because it finds the worst-case perturbation within the allowed budget.

**Q3: Why does adversarial training work?**  
By including adversarial examples in training batches, the model learns decision boundaries that are robust to small perturbations. The model's loss landscape becomes flatter near data points.

**Q4: What is the evasion rate?**  
The percentage of fraud transactions that successfully fool the model into predicting them as normal after the attack is applied.

**Q5: What are limitations of feature squeezing?**  
It may reduce accuracy on clean data slightly. A sophisticated attacker can craft perturbations that survive quantisation. It is a preprocessing defense, not a model-level fix.

**Q6: Why does class imbalance matter here?**  
Only 0.17% of transactions are fraud. Without SMOTE oversampling, the model learns to always predict "normal" and achieves 99.83% accuracy while missing all fraud — useless in practice.

**Q7: What is ε (epsilon)?**  
The maximum allowed perturbation per feature — the attacker's "budget." Small ε = subtle attack, hard to detect but may not fool the model. Large ε = obvious attack, more likely to fool but detectable.

**Q8: How does SMOTE work?**  
SMOTE (Synthetic Minority Oversampling Technique) creates synthetic fraud examples by interpolating between existing fraud samples in feature space, giving the model balanced training data.

**Q9: What real-world implications does this have?**  
Banks and fintech companies must harden ML models against adversarial manipulation. An attacker with knowledge of the model (white-box) or just its behaviour (black-box) could systematically bypass automated fraud checks.

**Q10: What is the clip_values parameter in ART?**  
It constrains adversarial examples to stay within a realistic feature range (e.g. transaction amounts can't be negative). This ensures generated attacks remain physically plausible.

---

## 📚 References

1. Goodfellow et al. (2014). *Explaining and Harnessing Adversarial Examples*. ICLR 2015.
2. Madry et al. (2017). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR 2018.
3. Xu et al. (2017). *Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks*.
4. Nicolae et al. (2018). *Adversarial Robustness Toolbox v1.0.0*. arXiv:1807.01069.
5. Dal Pozzolo et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification* (Kaggle dataset).

---

*Built for AI Security & Cryptography coursework. For academic use only.*
