from .token import Token
from abc import abstractmethod


class Variable(Token):

    @property
    @abstractmethod
    def slug(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def leftbp(self):
        raise NotImplementedError()

    def nud(self):
        return self

    def __repr__(self):
        return "(Variable %s)" % self.slug
