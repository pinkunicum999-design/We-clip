# ==============================
# AUTO CLIPPER - SAAS READY BACKEND (FULL VERSION)
# ==============================

import os
import uuid
import shutil
from datetime import datetime
from typing import Generator

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# DATABASE
# ==============================

DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String)
    status = Column(String, default="pending")
    result = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    plan = Column(String, default="free")


Base.metadata.create_all(bind=engine)

# ==============================
# FASTAPI INIT
# ==============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# DEPENDENCIES
# ==============================


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==============================
# UTIL
# ==============================


def generate_id():
    return uuid.uuid4().hex

# ==============================
# MOCK AUTH (REPLACE LATER)
# ==============================


def get_current_user(db: Session):
    user_id = "demo_user"

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        user = User(id=user_id, plan="free")
        db.add(user)
        db.commit()

    return user

# ==============================
# CORE PROCESS (REPLACE WITH YOUR AI PIPELINE)
# ==============================


def process_video(job_id: str, video_path: str):
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()

    try:
        job.status = "processing"
        db.commit()

        # ===== REPLACE THIS WITH YOUR REAL PIPELINE =====
        import time
        time.sleep(5)

        output_file = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
        shutil.copy(video_path, output_file)
        # ===============================================

        job.status = "done"
        job.result = output_file
        db.commit()

    except Exception as e:
        job.status = "error"
        job.result = str(e)
        db.commit()

    finally:
        db.close()

# ==============================
# ROUTES
# ==============================

@app.get("/")
def root():
    return {"status": "running"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_id = generate_id()
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return {"file_path": file_path}


@app.post("/generate")
def generate_video(
    file_path: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    user = get_current_user(db)

    # ===== PLAN LIMIT =====
    if user.plan == "free":
        # Example restriction (you can expand later)
        pass

    job_id = generate_id()

    job = Job(
        id=job_id,
        user_id=user.id,
        status="pending"
    )

    db.add(job)
    db.commit()

    background_tasks.add_task(process_video, job_id, file_path)

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "status": job.status,
        "result": job.result
    }


@app.get("/result/{job_id}")
def get_result(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job or job.status != "done":
        raise HTTPException(status_code=404, detail="Result not ready")

    return {"download_url": job.result}


@app.post("/upgrade")
def upgrade_plan(db: Session = Depends(get_db)):
    user = get_current_user(db)

    user.plan = "pro"
    db.commit()

    return {"message": "Upgraded to pro"}
