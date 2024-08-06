from typing import List

from colossalai.booster import Booster

from .state import TrainingState


def save_checkpoint() -> None:
    raise NotImplementedError()


def load_checkpoint(booster: Booster, checkpoint_dir: str) -> None:
    raise NotImplementedError()


class Checkpointer:

    def __init__(self) -> None:
        pass

    def save_checkpoint(self) -> None:
        raise NotImplementedError()

    def load_checkpoint(self) -> None:
        raise NotImplementedError()

    @classmethod
    def sorted_checkpoints(cls) -> List[str]:
        raise NotImplementedError()

    @classmethod
    def rotate_checkpoints(cls) -> None:
        raise NotImplementedError()

    @classmethod
    def save_rng_state(cls, output_dir: str, training_state: TrainingState) -> None:
        raise NotImplementedError()

    @classmethod
    def load_rng_state(cls, checkpoint_dir: str) -> None:
        raise NotImplementedError()

    @classmethod
    def save_training_state(cls) -> None:
        raise NotImplementedError()


