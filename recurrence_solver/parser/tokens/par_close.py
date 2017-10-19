from .token import Token


class ParenthesisClose(Token):

    slug = ")"
    leftbp = 0

    def __repr__(self):
        return "(Parenthesis close)"
