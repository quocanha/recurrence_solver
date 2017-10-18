from .tokenizer import Tokenizer


class Expression:
    """
    The parser works by the Top Down Operator precedence algorithm by Vaughan
    Pratt.

    It works by evaluating a token stream, generated by the Tokenize class.
    Each of the tokens created by this Tokenize class has a left binding power
    property. We then parse these symbols from left to right.

    When evaluating a token, we have a right binding power that is dynamic with
    which we compare to the left binding power of the next token in the
    expression.

    If the next token has a lower left binding power than the
    current right binding power, we return the current left value.

    However, if the next token's left binding power is greater than the current
    right binding power, that means we have to evaluate the next one first.

    We then start evaluating that next token by calling the led
    (left denotation) function. This functions takes as argument the, already
    evaluated, left, or previous symbols. This led function, defined in the
    token, then calls upon the same expression.parse function again with its
    right binding power, to run the same evaluation again for the next symbol.

    This is a recurrence that happens until the next symbol has a lower
    left binding power than the current right binding power. Which starts the
    ascend from the recursion, going through the branching logic of every
    led call of the pending symbols with a higher binding power.

    When having ascended from a branch back to the  root level, the initial
    recursion call will then continue with the rest of the tokens that
    are not processed by the first branch.

    This means that a symbol or operator is left associated, if the call to
    the expression.parse function in the led function, is done with a right
    binding power equal or less than the symbols left binding power.

    The nud function of a symbol will only be called once when the
    expression.parse function is called. This function in turn is only called
    from within a led function. This means that a syntax error by two sequential
    operators will result in a call to the nud function on the operator.

    """

    next = None
    next_token = None

    def __init__(self, raw):
        tokenizer = Tokenizer(self)
        generator = tokenizer.tokenize(raw)
        self.next = generator.__next__
        self.next_token = self.next()

    def parse(self, rightbp=0):
        """
        This function is the core of the parser.

        :param rightbp:
        :return:
        """
        current_token = self.next_token
        self.next_token = self.next()
        left = current_token.nud()

        while rightbp < self.next_token.leftbp:
            current_token = self.next_token
            self.next_token = self.next()
            left = current_token.led(left)

        return left
