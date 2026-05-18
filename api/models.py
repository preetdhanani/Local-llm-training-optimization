import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON
from .database import Base

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(String, default="PENDING")  # PENDING, RUNNING, COMPLETED, FAILED
    type = Column(String)  # SFT, DPO, FULL
    config = Column(JSON)
    pid = Column(Integer, nullable=True)
    log_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
