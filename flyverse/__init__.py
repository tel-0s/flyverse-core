"""flyverse: put the MaleCNS v1.0 fly connectome into simulations, games and other strange places."""

__all__ = ["FlyBrain", "MotorRates", "StepResult", "AsyncFlyBrain"]


def __getattr__(name):
    # Keep CLI tools such as python -m flyverse.connectome free of eager torch imports.
    if name in ("FlyBrain", "StepResult"):
        from . import fly
        return getattr(fly, name)
    if name == "MotorRates":
        from .motor import MotorRates
        return MotorRates
    if name == "AsyncFlyBrain":
        from .async_brain import AsyncFlyBrain
        return AsyncFlyBrain
    raise AttributeError(name)
