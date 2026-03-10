from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    checkpoint_dir: str = "data/checkpoints"
    synthetic_dir: str = "data/synthetic"
    real_dir: str = "data/real"

    image_size: int = 64
    latent_dim: int = 100
    num_classes: int = 2
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 0.0002

    app_env: str = "development"

    class Config:
        env_file = ".env"


settings = Settings()
