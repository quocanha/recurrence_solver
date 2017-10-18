from abc import ABC


class Token(ABC):
    def __init__(self, expression):
        self.expression = expression

    @property
    def leftbp(self):
        raise NotImplementedError()

    @property
    def slug(self):
        raise NotImplementedError()