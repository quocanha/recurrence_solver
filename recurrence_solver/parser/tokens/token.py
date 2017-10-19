from abc import ABC, abstractmethod


class Token(ABC):
    def __init__(self, expression):
        self.expression = expression

    @property
    @abstractmethod
    def leftbp(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def slug(self):
        raise NotImplementedError()
