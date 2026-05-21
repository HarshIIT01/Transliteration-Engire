# Evaluation Metrics

## Why These Metrics?

Transliteration is **not** a classification task — it's a sequence generation task where partial correctness matters. A model that predicts "नमस्त" instead of "नमस्ते" is much better than one that predicts "कुत्ता", even though both are technically "wrong". So we need metrics that measure **how wrong**, not just **if wrong**.

---

## Training Metrics

### 1. Cross-Entropy Loss

**What:** Measures how far the model's predicted probability distribution is from the true next character at each decoding step.

**Why:** This is the direct optimization target. The model is trained to minimize this loss, which pushes it to assign high probability to the correct Devanagari character at every position.

**Formula:**
```
Loss = -Σ log P(correct_char_t | input, chars_1..t-1)
```

**How to read it:**
- Lower = better
- Train loss decreasing = model is learning
- Val loss decreasing = model is generalizing
- Val loss increasing while train loss decreases = overfitting

---

### 2. Perplexity (PPL)

**What:** Exponential of cross-entropy loss. Intuitively, it represents **how many characters the model is "confused" between** at each step.

**Why:** More interpretable than raw loss. A perplexity of 5 means the model is roughly as uncertain as if it were choosing uniformly between 5 characters.

**Formula:**
```
PPL = e^(Cross-Entropy Loss)
```

**How to read it:**
- PPL = 1 → perfect (zero uncertainty)
- PPL = 5 → choosing between ~5 equally likely characters
- PPL = 70+ → model hasn't learned meaningful patterns yet
- Our target: PPL < 5 on validation set

---

### 3. Teacher Forcing Ratio

**What:** Probability that the decoder receives the **ground truth** previous character vs its **own prediction** during training.

**Why:** We track this because it follows a decay schedule:
- **High TF (0.7):** Early training — model learns with "training wheels"
- **Low TF (0.0):** Late training — model learns to recover from its own mistakes

**How to read it:**
- If val loss spikes when TF drops → model is too dependent on ground truth
- Smooth transition → healthy curriculum

---

## Evaluation Metrics (Post-Training)

### 4. Character Error Rate (CER)

**What:** Edit distance between predicted and reference Devanagari strings, normalized by reference length. Measures character-level accuracy.

**Why:** This is the **primary metric** for transliteration quality. Unlike word-level metrics, CER gives credit for partially correct outputs.

**Formula:**
```
CER = Levenshtein(predicted, reference) / len(reference)
```

**Examples:**
| Predicted | Reference | Edit Dist | CER |
|-----------|-----------|-----------|-----|
| नमस्ते | नमस्ते | 0 | 0.00 |
| नमस्त | नमस्ते | 1 | 0.17 |
| कुत्ता | नमस्ते | 6 | 1.00 |

**How to read it:**
- CER = 0 → perfect match
- CER < 0.1 → excellent (most characters correct)
- CER < 0.3 → acceptable for real-world use
- CER > 0.5 → poor quality

---

### 5. Word Error Rate (WER)

**What:** Edit distance between predicted and reference at the **word level**, normalized by reference word count. Relevant for sentence-level transliteration.

**Why:** CER alone can be misleading for sentences. A model might get individual characters right but split/merge words incorrectly. WER catches this.

**Formula:**
```
WER = WordEditDistance(predicted_words, reference_words) / len(reference_words)
```

**How to read it:**
- WER = 0 → every word matches perfectly
- WER = 0.2 → 80% of words are correct
- For our Stage 2 (word-level) model, WER ≈ 1 - ExactMatchAccuracy

---

### 6. Exact Match Accuracy (Top-1)

**What:** Percentage of inputs where the predicted output **exactly equals** the reference.

**Why:** The strictest metric — no partial credit. Important for user-facing quality: a keyboard that gets 85% of words exactly right feels usable; 60% feels broken.

**Formula:**
```
Accuracy = (# exact matches) / (# total samples)
```

**How to read it:**
- 90%+ → production quality
- 70-90% → good, usable with dictionary backoff
- < 70% → needs more training or data

---

## Why This Combination?

| Metric | Granularity | What it catches |
|--------|-------------|-----------------|
| Loss / PPL | Per-character (training) | Learning progress, overfitting |
| CER | Per-character (eval) | Partial correctness, near-misses |
| WER | Per-word | Word boundary errors in sentences |
| Exact Match | Per-sample | Strict real-world usability |

Using all four together gives a complete picture:
- **Loss/PPL** tell us *if the model is learning*
- **CER** tells us *how close the outputs are*
- **WER** tells us *if sentences are coherent*
- **Exact Match** tells us *if it's good enough to ship*

---

## Edge Deployment Metric

### 7. Inference Latency

**What:** Wall-clock time per sentence in milliseconds.

**Why:** For a keyboard use case, the model must respond in real-time. We benchmark on CPU (simulating mobile) and target **< 50ms per sentence**.

| Model Variant | Expected Latency |
|---------------|-----------------|
| PyTorch FP32 | ~20-40ms |
| Quantized INT8 | ~10-25ms |
| ONNX Runtime | ~5-15ms |
