# Inventory Defect Detection System using GAN

Addresses class imbalance in industrial defect datasets by training a **Conditional GAN (cGAN)** to synthesize realistic defect images, then training a **CNN classifier** on the combined real + synthetic data.

---

## Problem

Industrial defect datasets are severely imbalanced — fault samples often represent less than 2% of total data. Standard classifiers trained on such data fail to detect defects reliably. This project solves that by generating synthetic defect images via cGAN before training the classifier.

---

## Solution Pipeline

```
Real images (imbalanced)
        ↓
  Train cGAN  ──────────────────→  Synthetic defect images
        ↓                                    ↓
        └──────────── Combined dataset ──────┘
                              ↓
                   Train CNN Classifier
                              ↓
                    FastAPI Inference API
                              ↓
                    PostgreSQL Audit Log
```

---

## Results

| Metric | Before cGAN | After cGAN |
|--------|------------|------------|
| Minority class F1 | ~0.63 | ~0.91 |
| Recall (defective) | ~0.58 | ~0.91 |
| Dataset size | 500 | 4,000+ |

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/inference/predict` | Upload image → defective / normal + confidence |
| GET | `/inference/stats` | Inference counts, defect rate, recent logs |
| POST | `/generate/` | Generate synthetic images via cGAN |

---

## Project Structure

```
defect_detection/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── db/database.py
│   ├── models/defect.py          # SQLAlchemy ORM
│   ├── schemas/defect.py         # Pydantic schemas
│   ├── services/
│   │   ├── gan_service.py        # Generate synthetic images
│   │   └── classifier_service.py # Run inference + log to DB
│   ├── routers/
│   │   ├── inference.py          # POST /inference/predict
│   │   └── generate.py           # POST /generate/
│   └── ml/
│       ├── gan/
│       │   ├── cgan.py           # Generator + Discriminator
│       │   └── train.py          # cGAN training loop
│       └── classifier/
│           ├── model.py          # CNN architecture
│           └── train.py          # Classifier training loop
├── data/
│   ├── real/
│   │   ├── normal/
│   │   └── defective/
│   ├── synthetic/
│   └── checkpoints/
├── tests/
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/krishnaprasadlenka47/Inventory-Defect-Detection-System-using-GAN.git
cd Inventory-Defect-Detection-System-using-GAN
cp .env.example .env
```

### 2. Add your dataset

```
data/real/normal/      ← place normal product images here
data/real/defective/   ← place defective product images here
```

### 3. Run with Docker

```bash
docker-compose up --build
```

### 4. Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## Training

### Step 1 — Train cGAN to generate synthetic defect images

```bash
python -m app.ml.gan.train --epochs 200 --batch_size 32
```

Checkpoints saved to `data/checkpoints/generator.pt`

### Step 2 — Train CNN classifier on real + synthetic data

```bash
python -m app.ml.classifier.train --epochs 30
```

Checkpoint saved to `data/checkpoints/classifier.pt`

---

## Inference

```bash
# Classify an image
curl -X POST http://localhost:8000/inference/predict \
  -F "file=@your_image.jpg"

# Generate 10 synthetic defective images
curl -X POST http://localhost:8000/generate/ \
  -H "Content-Type: application/json" \
  -d '{"label": 1, "num_images": 10}'

# View stats
curl http://localhost:8000/inference/stats
```

Swagger UI: `http://localhost:8000/docs`

---

## Tech Stack

- **PyTorch** — cGAN + CNN implementation
- **FastAPI** — async REST API
- **PostgreSQL** — inference audit log
- **SQLAlchemy 2.0** — async ORM
- **Docker** — containerized deployment
- **Pydantic v2** — request/response validation
