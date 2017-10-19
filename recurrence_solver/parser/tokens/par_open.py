from .token import Token


class ParenthesisOpen(Token):

    slug = "("
    leftbp = 0

    first = None

    def nud(self):
        self.first = self.expression.parse()
        self.expression.advance(")")
        return self

    def __repr__(self):
        return "(Parenthesis (%s))" % self.first
