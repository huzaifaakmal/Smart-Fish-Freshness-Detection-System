# Fish Species Classification System

An AI-powered system designed to detect and classify fish species using the QUT Fish Dataset. This platform provides a complete end-to-end solution from data processing and model training to a real-time web dashboard and API.

## ⚠️ Important Disclaimer
**This system is strictly for Fish Species Classification.**
The QUT Fish Dataset does NOT contain freshness, spoilage, or quality labels. Therefore, this system does NOT detect fish freshness. Any future integration of freshness detection will require additional datasets.

---

## 🚀 Features
- **Species Detection:** Classifies fish into 297+ species (from the QUT dataset).
- **Environment Robustness:** Evaluates and handles various image conditions (Controlled, In-Situ, Sketches).
- **Deep Learning Backbone:** Utilizes EfficientNetB3 with Transfer Learning for high accuracy.
- **Real-time Inference:** Fast inference engine for single and batch image processing.
- **Interactive Dashboard:** Modern React-based web interface for easy uploading and result visualization.
- **Scalable Backend:** FastAPI-powered backend with database logging (PostgreSQL/SQLite).
- **Dockerized Deployment:** Ready for production deployment using Docker and Docker Compose.

---

## 🛠 Architecture
- **ML Framework:** PyTorch
- **Backend:** FastAPI (Python)
- **Frontend:** Next.js (React) / Tailwind CSS
- **Database:** PostgreSQL (Production) / SQLite (Local)
- **Deployment:** Docker / Docker Compose

---

## 📂 Project Structure
```text
project/
├── dataset/            # Data loading, parsing, and preprocessing
├── models/             # Model architectures and inference logic
├── training/           # Training loops and evaluation scripts
├── backend/            # FastAPI routes, database models, and schemas
├── frontend/           # Next.js web dashboard
├── docker/             # Dockerfiles and deployment configs
├── notebooks/          # EDA and analysis results
├── configs/            # Training and system configurations
└── utils/              # Shared helper functions
```

---

## 📥 Setup and Installation

### 1. Prerequisites
- Python 3.11+
- Docker & Docker Compose (Optional, for full stack)
- QUT Fish Dataset (should be placed in `Fish_Data/`)

### 2. Local Installation
```bash
# Clone the repository
git clone <repo-url>
cd smart-fish-freshness-detection-system

# Install dependencies
pip install -r requirements.txt

# Parse the dataset
python dataset/parser.py
```

### 3. Training the Model
```bash
python training/train.py
```
*Checkpoints will be saved in `models/checkpoints/best_model.pth`.*

### 4. Running the API
```bash
# Set environment variables (optional)
# export DATABASE_URL=sqlite:///./fish_app.db

python backend/main.py
```

### 5. Running the Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🐳 Docker Deployment
To run the entire system (Database + Backend + Frontend) using Docker:
```bash
docker-compose up --build
```

---

## 📊 Evaluation Metrics
The system achieved the following performance on the test set:
- **Top-1 Accuracy:** 71.14% (Correct species is the highest probability)
- **Top-5 Accuracy:** 91.25% (Correct species is within the top 5 probabilities)
- **Environment Robustness:** High performance maintained across In-Situ and Controlled conditions.

---

## 📝 Dataset Information
- **Source:** QUT Fish Dataset
- **Total Samples:** 4,411
- **Total Species:** 483 (Filtered to 297 for training)
- **Environments:** Controlled, In-Situ, Sketches, Uncontrolled.

---

## 🔮 Future Work
- Integration of actual freshness detection datasets.
- Support for fish size and weight estimation.
- Mobile application for field-based identification.
- Edge deployment for real-time cameras.
