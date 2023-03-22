from pydantic import BaseModel, Field


class EnvironmentConfig(BaseModel):
    board_width: float = Field(default=10.0)
    one_height: float = Field(default=10.0)
    two_height: float = Field(default=2.0)
    three_height: float = Field(default=2.0)
    
    num_turns: int = Field(default=2)
    puck_radius: float = Field(default=1.0)
    friction_coef: float = Field(default=0.9)
    max_agent_magnitude: float = Field(default=20.0)

    simulation_h: float = Field(default=0.5)
    max_time_steps: int = Field(default=100)
    min_velocity: float = Field(default=0.05)


class Optimization(BaseModel):
    gamma: float = Field(default=0.2)
    quality_lr: float = Field(default=0.01)
    actor_lr: float = Field(default=0.01)


class Config(BaseModel):
    environment: EnvironmentConfig = Field(default=EnvironmentConfig())
    optimization: Optimization = Field(default=Optimization())

    device: str = Field(default="cpu")
