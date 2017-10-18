from .token import Token
from abc import abstractmethod


class Atom(Token):

    value = None

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
    def value(self):
        raise NotImplementedError()

    def __repr__(self):
        return "(%s %s)" % (self.slug, self.value)
