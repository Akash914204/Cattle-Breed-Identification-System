# 🐄 BreedSense AI — Cattle Breed Recognition System

> An AI-powered web application that identifies **Indian cattle and buffalo breeds** from a single photo with high confidence, delivering real-time predictions alongside explainable Grad-CAM heatmaps — all backed by a secure, authenticated user portal.

---

## ✨ Key Features

- **Deep Learning Inference** — EfficientNetV2-S backbone fine-tuned on **50 Indian cattle and buffalo breeds** (cows and buffalos), delivering top-class predictions in milliseconds.
- **Explainable AI (XAI) with Grad-CAM** — Automatically generates a gradient-weighted class activation map (Grad-CAM) visualisation for every prediction, highlighting exactly which regions of the image drove the model's decision.
- **User Authentication** — Full register / login / logout flow with `bcrypt`-hashed passwords, session management via `Flask-Login`, and protected routes.
- **Prediction History** — Every prediction is persisted to MongoDB (breed, confidence score, upload filename, heatmap path, timestamp) and displayed per user in a history dashboard.
- **REST-ful Prediction API** — A JSON endpoint (`POST /api/predict`) accepts image uploads and returns breed name, confidence %, and a heatmap URL, making the core engine easy to integrate with other clients.
- **16 MB Upload Guard** — File-type validation (PNG / JPG / JPEG / WebP) and size limits are enforced server-side.
- **CUDA / CPU Agnostic** — Inference automatically uses a GPU when available and falls back to CPU transparently.
- **Multi-Environment Configuration** — Separate `development` and `production` config classes with environment-variable overrides via a `.env` file.

---

## 🛠️ Built With

| Category | Technologies |
|---|---|
| **Language** | Python 3.11+ |
| **Web Framework** | Flask 3.1, Flask-Login 0.6, Flask-CORS 5.0 |
| **Deep Learning** | PyTorch 2.6, TorchVision 0.21 |
| **Explainability** | Custom Grad-CAM implementation (Selvaraju et al., ICCV 2017) |
| **Database** | MongoDB (via PyMongo 4.11) |
| **Image Processing** | Pillow 11.1, NumPy 2.2, Matplotlib 3.10 |
| **Security** | bcrypt 4.2, python-dotenv 1.0 |
| **Templating** | Jinja2 (bundled with Flask) |
| **Production Server** | Gunicorn 23.0 |

---

## 🏗️ Architecture

The project follows a **Flask Application Factory** pattern with a clear separation of concerns across four layers:

```
Request → Auth / API Routes (Blueprints)
                ↓
         Service Layer  (PredictionService · GradCAM)
                ↓
         Model Layer    (EfficientNetV2-S weights + class list)
                ↓
         Data Layer     (MongoDB  — users & predictions collections)
```

- **Routes (Blueprints)** — Three Flask Blueprints (`main_bp`, `auth_bp`, `predict_bp`) keep routing logic isolated. The prediction API lives under the `/api` prefix.
- **Service Layer** — `PredictionService` owns model loading and standard inference; `GradCAMService` runs a separate forward–backward pass to produce the heatmap overlay image.
- **Model Layer** — Weights are loaded from `models/weights/best_model.pth`. The class roster is read from `models/classes.txt` at startup. The loader auto-detects whether the checkpoint was saved from a wrapper class (stripping the `backbone.` prefix) and reconstructs any custom multi-layer classifier head from weight shapes.
- **Data Layer** — MongoDB stores user documents and a `predictions` collection keyed by `user_id`.
- **Static assets** — Uploaded images land in `static/uploads/`; generated heatmap PNGs are written to `static/heatmaps/`.

---

## 🚀 Getting Started

### Prerequisites

| Tool | Minimum Version | Notes |
|---|---|---|
| Python | 3.11 | Earlier 3.x may work but is untested |
| MongoDB | 6.0 | Running locally on port `27017` or via Atlas |
| Git | any | For cloning |
| CUDA Toolkit *(optional)* | 11.8+ | Only required for GPU-accelerated inference |

---

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/akshat-2411/BreedSenseAI.git
cd BreedSenseAI

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

### ⚙️ Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env   # or edit the existing .env directly
```

Open `.env` and set the following variables:

```dotenv
# ── Flask ────────────────────────────────────────────────────
FLASK_APP=app.py
FLASK_ENV=development       # change to "production" for deployment
FLASK_DEBUG=1
SECRET_KEY=your-super-secret-key-here

# ── MongoDB ──────────────────────────────────────────────────
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=cattle_breed_db

# ── Model ────────────────────────────────────────────────────
MODEL_PATH=models/weights/best_model.pth
NUM_CLASSES=50
```

> **Note:** Place your trained model checkpoint at `models/weights/best_model.pth` before starting the server. The application will start without the weights file, but predictions will always return untrained outputs.

---

### ▶️ Running the App

**Development server (hot-reload):**

```bash
flask run
# or equivalently
python app.py
```

The server starts at **http://localhost:5000** by default.

**Production server (Gunicorn):**

```bash
gunicorn -w 4 -b 0.0.0.0:5000 "app:create_app()"
```

**CLI inference (standalone, no server needed):**

```bash
python inference.py path/to/cattle_image.jpg
```

---

## 📁 Project Structure

```
breedsense-ai/
│
├── app.py                    # Application factory — entry point
├── config.py                 # DevelopmentConfig / ProductionConfig
├── inference.py              # Standalone CLI inference script
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not committed)
│
├── models/
│   ├── classes.txt           # 50 breed names (one per line)
│   ├── user.py               # User model (Flask-Login integration)
│   └── weights/
│       └── best_model.pth    # Trained EfficientNetV2-S checkpoint
│
├── routes/
│   ├── auth_routes.py        # /auth/register  /auth/login  /auth/logout
│   ├── main_routes.py        # / (index), /about, /history
│   └── predict_routes.py     # POST /api/predict
│
├── services/
│   ├── prediction_service.py # Model loading + softmax inference
│   └── gradcam_service.py    # Grad-CAM heatmap generation & overlay
│
├── templates/
│   ├── base.html             # Shared layout / navigation
│   ├── index.html            # Landing page
│   ├── upload.html           # Image upload + results UI
│   ├── history.html          # Per-user prediction history
│   ├── about.html            # About page
│   └── auth/
│       ├── login.html
│       └── register.html
│
├── static/
│   ├── css/                  # Stylesheets
│   ├── js/                   # Client-side scripts
│   ├── uploads/              # User-uploaded images (auto-created)
│   └── heatmaps/             # Generated Grad-CAM overlays (auto-created)
│
├── utils/
│   └── helpers.py            # Shared utility functions
│
├── notebooks/
│   └── cattle_classifier_model.ipynb   # Training notebook
│
└── tests/                    # Test suite
```

---

## 🔌 API Reference

### `POST /api/predict`

Upload a cattle image to receive a breed prediction and Grad-CAM visualisation.

> **Authentication required.** Must be logged in.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `image` | file | JPG, PNG, JPEG, or WebP. Max 16 MB. |

**Response** — `200 OK`

```json
{
  "breed": "Gir Cow",
  "confidence": 94.37,
  "heatmap_url": "/static/heatmaps/gradcam_3f2a1b9c4d.png"
}
```

**Error Responses**

| Code | Meaning |
|---|---|
| `400` | Missing image or unsupported file type |
| `401` | Not authenticated |
| `500` | Inference failed server-side |

---

## 🌿 Supported Breeds (sample)

The model recognises **50 Indian cattle and buffalo breeds**, including:

`Gir Cow` · `Sahiwal Cow` · `Kankrej Cow` · `Tharparkar Cow` · `Rathi Cow` · `Deoni Cow` · `Dangi Cow` · `Hallikar Cow` · `Kangayam Cow` · `Nagori Cow` · `Jaffrabadi Buffalo` · `Banni Buffalo` · `Mehsana Buffalo` · `Nagpuri Buffalo` · `Nili Ravi Buffalo` · *and 35 more…*

See [`models/classes.txt`](models/classes.txt) for the complete list.

---

## 📄 License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

---

*Built with ❤️ for the Cattle Breed Recognition Capstone Project — 2026*
