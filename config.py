from pydantic import BaseModel, Field


class EnvironmentConfig(BaseModel):
    pass


class Config(BaseModel):

    environment_config: EnvironmentConfig = Field(default=EnvironmentConfig())
