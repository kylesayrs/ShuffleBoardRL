from pydantic import BaseModel, Field


class EnvironmentConfig(BaseModel):
    board_width: float = Field(default=10.0)
    one_height: float = Field(default=10.0)
    two_height: float = Field(default=2.0)
    three_height: float = Field(default=2.0)
    
    num_turns: int = Field(default=3)
    puck_radius: float = Field(default=0.5)
    friction_coef: float = Field(default=0.9)
    max_agent_magnitude: float = Field(default=2.0)

    simulation_h: float = Field(default=2.5)
    max_time_steps: int = Field(default=500)
    min_velocity: float = Field(default=0.05)


class Optimization(BaseModel):
    gamma: float = Field(default=0.1)
    quality_lr: float = Field(default=5e-7)
    actor_lr: float = Field(default=5e-7)
    quality_momentum: float = Field(default=0.05)
    actor_momentum: float = Field(default=0.05)

    num_episodes: int = Field(default=25_000)
    episodes_per_cycle: int = Field(default=16)
    batches_per_cycle: int = Field(default=40)
    batch_size: int = Field(default=16)
    replay_buffer_size: int = Field(default=10_000)

    epsilon_min: float = Field(default=1.0)
    epsilon_max: float = Field(default=0.0)
    noise_factor: float = Field(default=0.05)


class Config(BaseModel):
    env: EnvironmentConfig = Field(default=EnvironmentConfig())
    optim: Optimization = Field(default=Optimization())

    verbosity: int = Field(default=3)
    logging_rate: int = Field(default=100)
    device: str = Field(default="cpu")
