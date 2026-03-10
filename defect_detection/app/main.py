from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db.database import init_db
from app.routers import inference, generate


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Inventory Defect Detection System",
    description=(
        "Conditional GAN (cGAN) for synthetic defect image generation to address "
        "class imbalance, combined with a CNN classifier for defect detection. "
        "Built with FastAPI and PostgreSQL."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(inference.router)
app.include_router(generate.router)


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Defect Detection API is running"}
