from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration sourced from environment variables."""

    model_path: Path = Field(
        default=Path("models/asl-sign-detector.pt"),
        description="Filesystem path to the YOLO weights trained on ASL lexicon.",
    )
    confidence_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum probability required for a prediction to be considered valid.",
    )
    smoothing_window: int = Field(
        default=5,
        ge=1,
        description="Number of most recent predictions used when smoothing noisy outputs.",
    )
    frame_interval_ms: int = Field(
        default=400,
        ge=100,
        description="Interval between frames sent from the browser in milliseconds.",
    )
    jpeg_quality: float = Field(
        default=0.4,
        ge=0.1,
        le=1.0,
        description="JPEG quality used when encoding frames in the browser.",
    )

    model_config = SettingsConfigDict(
        env_prefix="asl_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @field_validator("model_path", mode="before")
    def _expand_model_path(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value.expanduser().resolve()
        return Path(str(value)).expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached instance of application settings."""

    return Settings()
