"""FIXED — typed settings from environment (secrets live here, never in code/config)."""
from __future__ import annotations
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_api_key: str = ""
    wandb_api_key: str = ""
    class Config:
        env_file = ".env"

settings = Settings()  # import this; do not read os.environ elsewhere
