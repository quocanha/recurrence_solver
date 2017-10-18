import re
from .tokens import *


class Tokenizer:

    pattern = re.compile("\s*(?:(\d+)|([a-zA-Z]+)|([+\-*/^]))")
    symbols = {}

    def __init__(self, expression):
        self.expression = expression

        self.create_atom("end")
        self.create_atom("literal").nud = lambda self: self

        self.create_infix("+", 10)
        self.create_infix("-", 10)

        self.create_infix("*", 20)
        self.create_infix("/", 20)

        self.create_prefix("+", 999)
        self.create_prefix("-", 999)

        self.create_infix_r("^", 30)

    def tokenize(self, raw):
        """
        Returns a generator for the tokens corresponding to the equation. Uses
        the tokens available in the tokens dictionary.

        :param raw: The equation to evaluate.
        :return: A generator for the stream of tokens corresponding to the
                 equation.
        """
        matches = self.pattern.findall(raw)
        for lit, var, op in matches:
            if lit:
                symbol = self.symbols["literal"]
                tokn = symbol(self.expression)
                tokn.value = lit
                yield tokn
            elif op:
                symbol = self.symbols.get(op)
                if not symbol:
                    raise SyntaxError("Invalid operator (%s)" % op)
                # IDE says symbol is not callable, however it is.
                tokn = symbol(self.expression)
                yield tokn
        symbol = self.symbols["end"]
        yield symbol(self.expression)

    def create_symbol(self, symbol_type, slug, leftbp=0):
        """
        Creates a symbol by the given symbol type if it doesn't exists. Sets
        the symbol-slug to the name of the class for easy debugging.

        If the symbol already exists, it just updates the left binding power
        if the new binding power is higher.
        :param symbol_type: A class object to instantiate a token from.
        :param slug: The name and identifier for the token.
        :param leftbp: The left binding power.
        :return: The symbol.
        """
        if slug in self.symbols:
            # If the symbol already exists, just update the new binding power.
            symbol = self.symbols[slug]
            symbol.leftbp = max(symbol.leftbp, leftbp)
        else:
            symbol = symbol_type
            symbol.slug = slug
            symbol.leftbp = leftbp
            symbol.__name__ = "symbol-" + slug

            # Create a new factory if it doesn't exist.
            self.symbols[slug] = symbol
        return symbol

    """
    Symbol create helpers.
    """

    def create_atom(self, slug, leftbp=0):
        """
        Creates an atom symbol.
        :param slug:
        :param leftbp:
        :return:
        """
        class atom(Atom):
            """
            Placeholder class to circumvent the abstractmethod requirement
            of the abstract functions of the Atom class, since we are
            setting it dynamically.
            """
            value = None
            slug = None
            leftbp = None
            pass

        return self.create_symbol(atom, slug, leftbp)

    def create_operator(self, slug, leftbp=0):
        """
        Creates
        :param slug:
        :param leftbp:
        :return:
        """
        class op(Operator):
            """
            Placeholder class to circumvent the abstractmethod requirement
            of the abstract functions of the Operator class, since we are
            setting it dynamically.
            """
            slug = None
            leftbp = None
            first = None
            second = None
            pass

        return self.create_symbol(op, slug, leftbp)

    def create_infix(self, slug, leftbp):
        """
        Creates an left associated infix symbol.

        :param slug:
        :param leftbp:
        :return:
        """
        op = self.create_operator(slug, leftbp)

        def led(self, left):
            self.first = left
            self.second = self.expression.parse(self.leftbp)
            return self

        op.led = led
        return op

    def create_infix_r(self, slug, leftbp):
        """
        Creates a right associated infix symbol.

        :param slug:
        :param leftbp:
        :return:
        """
        op = self.create_operator(slug, leftbp)

        def led(self, left):
            """
            The led function for a typical right associated infix symbol.

            :param self:
            :param left:
            :return:
            """
            self.first = left
            self.second = self.expression.parse(self.leftbp - 1)
            return self

        op.led = led
        return op

    def create_prefix(self, slug, rightbp):
        """
        Creates a prefix symbol.

        :param slug: Slug of the symbol.
        :param rightbp: Right binding power.
        :return: The symbol with a newly appended nud function.
        """
        op = self.create_operator(slug)

        def nud(self):
            """
            A prefix symbol is characterized by it's nud function. It calls
            the next symbol, with an absurdly high right binding power. This
            forces the branch to quit on the next symbol.

            :param self:
            :return:
            """
            self.first = self.expression.parse(rightbp)
            self.second = None
            return self

        op.nud = nud
        return op

