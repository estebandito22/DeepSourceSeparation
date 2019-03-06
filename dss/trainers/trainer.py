"""Abstract class to train models."""

from abc import ABC, abstractmethod


class Trainer(ABC):

    """Abstract class for a model trainer."""

    @abstractmethod
    def __init__(self):
        """Initialize abstract trainer class."""

    @abstractmethod
    def fit(self):
        """Train model."""
        raise NotImplementedError("train is an abstract method.")

    @abstractmethod
    def predict(self):
        """Evaluate model."""
        raise NotImplementedError("evaluate is an abstract method.")

    @abstractmethod
    def score(self):
        """Score model."""
        raise NotImplementedError("score is an abstract method.")

    @abstractmethod
    def save(self):
        """Save model."""
        raise NotImplementedError("save is an abstract method.")

    @abstractmethod
    def load(self):
        """Load model."""
        raise NotImplementedError("load is an abstract method.")
