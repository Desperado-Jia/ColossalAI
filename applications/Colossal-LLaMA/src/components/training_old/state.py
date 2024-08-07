from pydantic import BaseModel, Field


class TrainingState(BaseModel):
    """A class used to represent the training state."""

    epoch: float = Field(
        default=0.0,
        description="Epoch of the training."
    )
    step: int = Field(
        default=0,
        description="Training steps."
    )

