# ASL Translator

Real-time American Sign Language translator using your webcam, MediaPipe hand tracking, and machine learning. Recognizes both **alphabet letters** (A-Z) and **common word signs** using one or both hands.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)

## How It Works

1. **Hand Detection** — MediaPipe detects up to 2 hands and extracts 21 3D landmark points per hand
2. **Feature Extraction** — Landmarks are normalized per-hand, plus relative hand positioning is encoded (130-dim vector)
3. **Classification** — A Random Forest classifier predicts which ASL sign you're showing
4. **Sentence Building** — Hold a sign steady to add it to the sentence automatically

## Recognized Signs

### Alphabet (26 letters)
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z

### Word Signs (12 words)
| Sign | Hands | Description |
|------|-------|-------------|
| I Love You | 1 | Thumb + index + pinky extended |
| Good | 1 | Thumbs up |
| Yes | 1 | Fist (S-hand) |
| No | 1 | Index + middle + thumb pinching |
| More | 2 | Both hands bunched (flat-O), tips together |
| Help | 2 | Fist on open palm |
| Book | 2 | Two open palms together |
| Stop | 2 | Hand blade hitting open palm |
| Play | 2 | Both hands Y-shape |
| Want | 2 | Both hands claw/spread shape |
| With | 2 | Two fists brought together |
| Same | 2 | Both index fingers pointing together |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Training Data & Train

**Option A — Use a real dataset (recommended):**

1. Go to [Kaggle ASL Landmarks](https://www.kaggle.com/datasets/jaisuryaprabu/sign-language-landmarks) (free account, sign in with Google)
2. Click "Download" (2.7 MB ZIP)
3. Extract the CSV file into the `downloads/` folder
4. Run:

```bash
python import_dataset.py
python train_model.py
```

**Option B — Quick start with synthetic data:**

```bash
python generate_dataset.py
python train_model.py
```

### 3. Run the Translator

```bash
python asl_translator.py
```

## Translator Controls

| Key        | Action                      |
| ---------- | --------------------------- |
| **SPACE**  | Add a space to the sentence |
| **BACKSPACE** | Delete last character/word |
| **C**      | Clear the entire sentence   |
| **ESC**    | Quit                        |

## Project Structure

```
asl-translator/
├── asl_translator.py      # Main real-time translator (both hands)
├── import_dataset.py       # Import real ASL datasets (Kaggle CSV or images)
├── collect_data.py         # Webcam data collector for custom training
├── generate_dataset.py     # Synthetic dataset generator (letters + words)
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── downloads/              # Place downloaded datasets here
├── data/                   # Training data (generated)
│   └── landmarks.csv
└── model/                  # Models (generated/downloaded)
    ├── asl_classifier.pkl
    └── hand_landmarker.task
```

## Improving Accuracy

Three ways to improve, from easiest to most effective:

### Import a Real Dataset (easiest big improvement)

Download the Kaggle landmarks CSV and run `python import_dataset.py`. This replaces synthetic letter data with real hand data from multiple people. Also supports importing image datasets — see `import_dataset.py` for details.

### Collect Your Own Data (best personal accuracy)

Record your own hands for a model tuned to you:

```bash
python collect_data.py
```

**Letter signs:** Press a letter key (A-Z) to record that sign.

**Word signs:** Press a number key:
| Key | Sign |
|-----|------|
| 1 | I Love You |
| 2 | Good |
| 3 | More |
| 4 | Help |
| 5 | Book |
| 6 | Stop |
| 7 | Play |
| 8 | Want |
| 9 | With |
| 0 | Same |

Press **SPACE** to stop recording, **ESC** to save and quit. Aim for 200+ samples per sign.

Then retrain:

```bash
python train_model.py
```

## Requirements

- Python 3.8+
- Webcam
- Windows / macOS / Linux
