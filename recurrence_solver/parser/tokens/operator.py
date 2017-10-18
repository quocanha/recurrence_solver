from .token import Token
from abc import abstractmethod


class Operator(Token):

    first = None
    second = None

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
    def first(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def second(self):
        raise NotImplementedError()

    def __repr__(self):
        return "(%s %s %s)" % (self.slug, self.first, self.second)
