from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import multiprocessing
import os
import re
import signal
import shutil

from . import models, schemas, database
from .database import engine, get_db
from .env_manager import check_environment
from src.config import TrainConfig

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="RLHF Dashboard API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:6767"],  # New dashboard port
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
    
    # Check for PAUSED signal
    log_dir = os.path.dirname(job.log_path)
    # Extract timestamp from log filename: full_pipeline_train_2026-05-18_18-00-13.log
    match = re.search(r"train_(.*)\.log", os.path.basename(job.log_path))
    paused_data = None
    if match:
        timestamp = match.group(1)
        signal_path = f"./logs/signals_{timestamp}/PAUSED"
        if os.path.exists(signal_path):
            with open(signal_path, "r") as f:
                paused_data = f.read()
    
    if not os.path.exists(job.log_path):
        return {"logs": "Log file not created yet...", "paused": False}
        
    try:
        with open(job.log_path, 'r') as f:
            # Return last 100 lines
            lines = f.readlines()
            return {
                "logs": "".join(lines[-100:]),
                "paused": paused_data is not None,
                "paused_stats": paused_data
            }
    except Exception as e:
        return {"logs": f"Error reading logs: {e}", "paused": False}

@app.post("/jobs/{job_id}/approve")
def approve_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job or not job.log_path: return {"status": "error"}
    
    match = re.search(r"train_(.*)\.log", os.path.basename(job.log_path))
    if match:
        timestamp = match.group(1)
        signal_dir = f"./logs/signals_{timestamp}"
        os.makedirs(signal_dir, exist_ok=True)
        with open(f"{signal_dir}/APPROVED", "w") as f: f.write("1")
        # Cleanup paused signal
        if os.path.exists(f"{signal_dir}/PAUSED"): os.remove(f"{signal_dir}/PAUSED")
        return {"status": "success"}
    return {"status": "no_timestamp_found"}

@app.post("/jobs/{job_id}/abort")
def abort_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job: return {"status": "error", "detail": "Job not found"}
    
    # 1. Kill the process if it's running
    if job.status == "RUNNING" and job.pid:
        try:
            # Use SIGTERM for graceful but firm stop
            os.kill(job.pid, signal.SIGTERM)
        except Exception as e:
            print(f"Failed to kill process {job.pid}: {e}")

    # 2. Set Status in DB
    job.status = "ABORTED"
    db.commit()

    # 3. Write signal file for the pipeline loop (if it's in a pause state)
    if job.log_path:
        match = re.search(r"train_(.*)\.log", os.path.basename(job.log_path))
        if match:
            timestamp = match.group(1)
            signal_dir = f"./logs/signals_{timestamp}"
            os.makedirs(signal_dir, exist_ok=True)
            with open(f"{signal_dir}/ABORTED", "w") as f: f.write("1")
    
    return {"status": "success"}

def perform_deep_cleanup(job, db):
    """Helper to kill process and delete all related files/folders."""
    try:
        # 1. Kill if running
        if job.status == "RUNNING" and job.pid:
            try:
                os.kill(job.pid, signal.SIGTERM)
            except:
                pass

        # 2. Extract timestamp for folder lookup
        timestamp = None
        if job.log_path:
            match = re.search(r"train_(.*)\.log", os.path.basename(job.log_path))
            if match:
                timestamp = match.group(1)

        # 3. Delete log file
        if job.log_path and os.path.exists(job.log_path):
            try:
                os.remove(job.log_path)
            except:
                pass

        # 4. Cleanup signals and outputs
        if timestamp:
            shutil.rmtree(f"./logs/signals_{timestamp}", ignore_errors=True)
            shutil.rmtree(f"./outputs/sft_output_{timestamp}", ignore_errors=True)
            shutil.rmtree(f"./outputs/dpo_output_{timestamp}", ignore_errors=True)

        # 5. Delete from DB
        db.delete(job)
        db.commit()
    except Exception as e:
        print(f"Cleanup error for job {job.id}: {e}")
        # Always try to delete from DB even if file cleanup fails
        try:
            db.delete(job)
            db.commit()
        except:
            pass

@app.delete("/jobs/{job_id}")
def delete_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status in ["RUNNING", "AWAITING_APPROVAL"]:
        raise HTTPException(status_code=400, detail="Cannot delete an active job. Please Cancel/Abort it first.")
        
    perform_deep_cleanup(job, db)
    return {"status": "success"}

@app.delete("/jobs")
def delete_all_jobs(db: Session = Depends(get_db)):
    # Only delete terminal jobs (COMPLETED, FAILED, ABORTED)
    jobs = db.query(models.Job).filter(models.Job.status.notin_(["RUNNING", "AWAITING_APPROVAL"])).all()
    count = 0
    for job in jobs:
        perform_deep_cleanup(job, db)
        count += 1
    return {"status": "success", "count": count}

def run_training_task(job_id: int, config_dict: dict):
    """
    Background worker that runs the actual training pipeline.
    """
    from .database import SessionLocal
    try:
        from src.pipelines.run_full_pipeline import run_full_pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are not installed in this image. Use the full training image or install requirements.txt."
        ) from exc

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
