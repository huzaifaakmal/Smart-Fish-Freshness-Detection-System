from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import shutil
import os
import uuid
from typing import List

from backend.database import engine, get_db
from backend.models import Base, Prediction, UploadedImage
from models.inference import FishInference

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Fish Species Classification API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize inference engine
# Note: In production, path would be loaded from env or config
MODEL_PATH = "models/checkpoints/best_model.pth"
inference_engine = None

def get_inference_engine():
    global inference_engine
    if inference_engine is None:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}...")
            inference_engine = FishInference(MODEL_PATH)
    return inference_engine

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fish Species Classification API", "status": "online"}

@app.get("/health")
def health_check():
    engine_ready = get_inference_engine() is not None
    return {
        "status": "healthy",
        "model_loaded": engine_ready,
        "database_type": str(engine.url).split(":")[0]
    }

@app.post("/predict")
async def predict_fish(file: UploadFile = File(...), db: Session = Depends(get_db)):
    engine = get_inference_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not loaded. Model training is still in progress. Please try again in a few minutes.")
        
    # Save uploaded file
    file_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1]
    filename = f"{file_id}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Run inference
    try:
        results = inference_engine.predict(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
    # Log to database
    db_prediction = Prediction(
        image_name=file.filename,
        predicted_species=results["predicted_species"],
        confidence=results["confidence"],
        environment_detected=None # Environment detection can be added to inference
    )
    db.add(db_prediction)
    
    db_upload = UploadedImage(
        filename=file.filename,
        file_path=file_path
    )
    db.add(db_upload)
    
    db.commit()
    
    return {
        "filename": file.filename,
        "prediction": results["predicted_species"],
        "confidence": results["confidence"],
        "top_5": results["top_5"],
        "disclaimer": "This system supports species classification only. Freshness detection is not supported."
    }

@app.get("/history")
def get_prediction_history(limit: int = 10, db: Session = Depends(get_db)):
    predictions = db.query(Prediction).order_by(Prediction.timestamp.desc()).limit(limit).all()
    return predictions

@app.get("/species")
def get_species_list(db: Session = Depends(get_db)):
    # In a real app, this would query a populated Species table
    if inference_engine:
        return {"species": list(inference_engine.idx_to_species.values())}
    return {"species": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
