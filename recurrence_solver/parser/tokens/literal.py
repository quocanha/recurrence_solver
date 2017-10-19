from .token import Token


class Literal(Token):

    value = None
    slug = None
    leftbp = None

    def nud(self):
        return self

    def __repr__(self):
        return "(Literal %s)" % self.value
