from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_key: str
    openrouter_model_list: list[str]

    @classmethod
    @field_validator("openrouter_model_list", mode="before")
    def validate_openrouter_model_list(cls, v):
        return [model.strip() for model in v.split(",")]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )


settings = Settings()  # type: ignore
