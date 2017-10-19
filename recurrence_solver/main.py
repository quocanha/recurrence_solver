from input.input import Input
from solver.solver import Solver
from parser.expression import Expression
from sympy import *


class Main:
    """Main class."""

    def __init__(
            self,
            input_folder="../instructions/input_example",
            output_folder="../instructions/output",
            debug=False
    ):
        self.input = Input(input_folder)
        self.output_folder = output_folder
        self.debug = debug

    def iterative(self):
        recurrences = self.input.recurrences
        for recurrence in recurrences:
            slver = Solver(recurrences[recurrence])
            slver.solve()

    def topdown(self):
        recurrences = self.input.recurrences
        for recurrence in recurrences:
            recurrence = recurrences[recurrence]
            recurrence.print()
            expr = Expression(recurrence.recurrence)
            tree = expr.parse()
            print(tree)
            print()

    def smpy(self):
        exp = sympify("-4*s( n-2) + 4* s ( n-1)")
        print(exp)

    def print(self):
        """ """
        recurrences = self.input.recurrences
        for recurrence in recurrences:
            print("File: " + recurrence)
            recurrences[recurrence].print()
            print("")


if __name__ == '__main__':
    solver = Main()
    solver.topdown()
