from input.input import Input
from solver.solver import Solver
from parser.expression import Expression


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
            # expr = Expression(recurrences[recurrence].recurrence)
            recurrences[recurrence].print()
            expr = Expression("1 + 2 * 3 / 4 ^ 5 ^ -5")
            print(expr.parse())
            print("")

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
