from datetime import datetime
from sqlalchemy import String, Float, Integer, Boolean, DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    predicted_class: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_defective: Mapped[bool] = mapped_column(Boolean, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_type: Mapped[str] = mapped_column(String(50), nullable=False)
    epochs_completed: Mapped[int] = mapped_column(Integer, nullable=False)
    final_g_loss: Mapped[float] = mapped_column(Float, nullable=True)
    final_d_loss: Mapped[float] = mapped_column(Float, nullable=True)
    final_accuracy: Mapped[float] = mapped_column(Float, nullable=True)
    checkpoint_path: Mapped[str] = mapped_column(String(500), nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
