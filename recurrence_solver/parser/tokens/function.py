from .token import Token
from abc import abstractmethod


class Function(Token):

    name = None
    first = None

    @property
    @abstractmethod
    def slug(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def leftbp(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError()

    def nud(self):
        self.first = self.expression.parse()
        self.expression.advance(")")
        return self

    def __repr__(self):
        return "(Function %s (%s))" % (self.name, self.first)
