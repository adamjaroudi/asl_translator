# Getting the correct letters

If the translator shows the **wrong letter** (e.g. you sign "A" but it shows "S"), the model is usually confused because it was trained on **synthetic** hand poses, not real camera images of hands.

---

## Best fix: train on real letter data

### Option A — Kaggle dataset (quick, big improvement)

1. Download the ASL landmarks CSV from **[Kaggle: Sign Language Landmarks](https://www.kaggle.com/datasets/jaisuryaprabu/sign-language-landmarks)** (free account).
2. Extract the CSV and put it in the **`downloads/`** folder in this project.
3. Run:
   ```bash
   python import_dataset.py
   python train_model.py --accuracy
   ```
4. Run the translator again: `python asl_translator.py`

That replaces synthetic letter data with **real** hand landmarks from many people, so letters usually match much better.

---

### Option B — Your own hand (best match to you)

1. Run the data collector:
   ```bash
   python collect_data.py
   ```
2. For each letter A–Z:
   - Press the **letter key** (e.g. **A**).
   - Hold the sign, move your hand slightly, then hold again (so you get variety).
   - Press **SPACE** to stop.
   - Aim for **150–300 samples per letter** (more = better).
3. When done, press **ESC** to save.
4. Retrain:
   ```bash
   python train_model.py --accuracy
   ```
5. Run: `python asl_translator.py`

The model will then be tuned to **your** hand and camera.

---

## What you’ll see in the app

- **“or: N? S?”** under the main letter means the model is unsure between those letters. Try holding the sign more clearly or improving lighting.
- In the **terminal**, when a letter is added you’ll see e.g. `Added: M (also: N 18%, S 6%)` so you can see which letters get confused.

---

## Summary

| Situation | What to do |
|----------|-------------|
| Wrong letter often | Use **real data**: `import_dataset.py` (Kaggle CSV in `downloads/`) then `train_model.py --accuracy` |
| Still wrong for some letters | **Collect your own** with `collect_data.py` for those letters, then retrain with `--accuracy` |
| “or: X? Y?” on screen | Model is unsure; hold the sign steadier and make it clearer |
