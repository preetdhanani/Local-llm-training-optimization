from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import multiprocessing
import time
import os
import logging

from . import models, schemas, database
from .database import engine, get_db
from .env_manager import check_environment
from src.config import TrainConfig
from src.pipelines.run_full_pipeline import run_full_pipeline

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="RLHF Dashboard API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/env-check", response_model=schemas.EnvResponse)
def get_env():
    return check_environment()

@app.get("/jobs", response_model=List[schemas.JobResponse])
def list_jobs(db: Session = Depends(get_db)):
    return db.query(models.Job).all()

@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: int, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job or not job.log_path:
        raise HTTPException(status_code=404, detail="Job or log path not found")
    
    if not os.path.exists(job.log_path):
        return {"logs": "Log file not created yet..."}
        
    try:
        with open(job.log_path, 'r') as f:
            # Return last 100 lines
            lines = f.readlines()
            return {"logs": "".join(lines[-100:])}
    except Exception as e:
        return {"logs": f"Error reading logs: {e}"}

def run_training_task(job_id: int, config_dict: dict):
    """
    Background worker that runs the actual training pipeline.
    """
    from .database import SessionLocal
    db = SessionLocal()
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    
    if not job:
        return

    try:
        # Update job with PID
        job.status = "RUNNING"
        job.pid = os.getpid()
        db.commit()

        # Convert dict to TrainConfig
        cfg = TrainConfig()
        for key, value in config_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        
        # Sanity Check for common UI/Cache errors
        if cfg.model_id == "HuggingFace" or cfg.model_id == "Local File":
             raise ValueError(f"Invalid Model ID detected: '{cfg.model_id}'. Please refresh your browser and enter a valid HuggingFace ID or local path.")
        
        # Enforce the strict standard
        cfg.dataset_type = "local"
        cfg.dataset_format = "custom_columns"
        cfg.prompt_column = "prompt"
        cfg.chosen_column = "chosen"
        cfg.rejected_column = "rejected"
        cfg.system_prompt = None # Rely on prompt column contents
        
        # We need to capture the log path.
        # run_full_pipeline sets up its own logger, but we can pre-calculate the path
        from src.utils.logger import get_timestamped_log_path
        log_path = get_timestamped_log_path("full_pipeline")
        job.log_path = log_path
        db.commit()

        # Run pipeline
        run_full_pipeline(cfg)
        
        job.status = "COMPLETED"
        db.commit()
    except Exception as e:
        # Simple logging in background process
        print(f"Background Training Error: {e}")
        job.status = "FAILED"
        db.commit()
    finally:
        db.close()

@app.post("/jobs/train", response_model=schemas.JobResponse)
def start_training(config: schemas.TrainingConfigSchema, db: Session = Depends(get_db)):
    # Create job entry
    job = models.Job(type="FULL_PIPELINE", config=config.model_dump())
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Start process-level isolation task
    process = multiprocessing.Process(
        target=run_training_task, 
        args=(job.id, config.model_dump())
    )
    process.start()
    
    return job

@app.post("/jobs/dummy", response_model=schemas.JobResponse)
def start_dummy_job(db: Session = Depends(get_db)):
    job = models.Job(type="DUMMY", config={"seconds": 10})
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Inline dummy for quick testing
    def dummy_logic(jid):
        from .database import SessionLocal
        d = SessionLocal()
        j = d.query(models.Job).filter(models.Job.id == jid).first()
        j.status = "RUNNING"
        j.pid = os.getpid()
        d.commit()
        time.sleep(10)
        j.status = "COMPLETED"
        d.commit()
        d.close()

    process = multiprocessing.Process(target=dummy_logic, args=(job.id,))
    process.start()
    
    return job
