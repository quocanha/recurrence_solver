from sympy import *

from input.input import Input
from interface.menu import Menu
from recurrence.recurrence import Recurrence
from solver.iterativesolver import IterativeSolver
from parser.expression import Expression


class Main:
    """Main class."""

    queue = {}

    def __init__(
            self,
            input_folder="../instructions/input_example",
            output_folder="../instructions/output",
            debug=False
    ):
        self.input = Input(input_folder)
        self.output_folder = output_folder
        self.debug = debug

        self.menu = Menu()

    def run(self):
        """ """
        self.menu.welcome()
        # selection = self.menu.input(self.input.recurrences)
        selection = 3

        if selection == 0:
            self.queue = self.input.recurrences
        else:
            filename = list(self.input.recurrences.keys())[selection - 1]
            self.queue[filename] = self.input.recurrences[filename]

        for filename in self.queue:
            recurrence = self.input.recurrences[filename]
            try:
                print("**** Recurrence ****")
                print(recurrence)
                print()
                hypothesis = recurrence.solve()
                print()
                print()
            except Exception as e:
                print(e)
                print()
                pass
            else:
                n = symbols("n", integer=True)
                solution = None
                if recurrence.filename == "comass16.txt":
                    solution = sympify("139/216*(-1)^n*2^n*n+1/9*n^3+161/162*(-1)^(n+1)*2^n+41/8*2^n*n+16/9*n^2-47*2^(n-1)+32/3*n+1984/81")
                elif recurrence.filename == "comass07.txt":
                    solution = sympify("1/10*(1/2*5^(1/2)+1/2)^n*5^(1/2)+1/2*(1/2*5^(1/2)+1/2)^n+1/2*(-1/2*5^(1/2)+1/2)^n-1/10*(-1/2*5^(1/2)+1/2)^n*5^(1/2)")
                if recurrence.filename == "comass03.txt":
                    solution = sympify("-2*2^n*(n-3)")
                if recurrence.filename == "comass36.txt":
                    solution = sympify("-71/1650*(-1)^n*3^n*n+5089/60500*(-1)^n*3^n-254/975*2^n*n+138484/190125*2^n+1/2944656*41^n+3/16")

                if solution is not None:
                    test = solve(Eq(simplify(hypothesis) - simplify(solution), 0), n)
                    if test is not True:
                        print("Test incorrect.")
                        i = 0
                        n = symbols("n")
                        correct = True
                        while i < 20:
                            r = simplify(hypothesis.subs(n, i))
                            solution_i = simplify(solution.subs(n, i))
                            print("\t\t{}\t\t{}".format(r, solution_i))
                            correct = solve(Eq(r, solution_i))
                            i = i + 1
                        if correct:
                            print("Iteration showed that it is correct.")
                    else:
                        print("Correct.")

    def working(self):
        recurrence = Recurrence(
            "eqs :=[s(n)=s(n-1) + n, s(1)=1]")
        recurrence.solve()

    def test2(self):
        recurrence = Recurrence(
            "eqs :=[s(n)=4*s(n-1) - 3*s(n-2) + 2^n + n + 3, s(0)=1, s(1)=4]")
        recurrence.solve()

    def test3(self):
        recurrence = Recurrence(
            "eqs :=[s(n)=s(n-1) + 3^n, s(0)=1, s(1)=2]")
        solution = recurrence.solve()

    def test5(self):
        recurrence = Recurrence("eqs :=[s(n)=s(n-2)+(1/2)*n^2+(1/2)*n,s(0)=0,s(1)=1]")
        recurrence.solve()

    def test6(self):
        recurrence = Recurrence("eqs :=[s(n)=8*s(n-1) + 10^(n-1), s(1)=9]")
        print(recurrence.solve())

    def cats(self):
        recurrence = Recurrence("eqs :=[s(n)=s(n-2)+(1/2)*n^2 + (1/2)*n, s(0)=0, s(1)=1]")
        print(recurrence)
        hypthesis = recurrence.solve()
        solution = sympify("(1/16) - (1/16) * (-1)^n + (1/12)*n^3 + (3/8)*n^2 + (5/12)*n")
        n = symbols("n")
        i = 0
        while i < 20:
            r = simplify(hypthesis.subs(n, i))
            solution_i = simplify(solution.subs(n, i))
            print("\t\t{}\t\t{}".format(r, solution_i))
            i = i + 1

    def iterative(self):
        recurrences = self.input.recurrences
        for recurrence in recurrences:
            slver = IterativeSolver(recurrences[recurrence])
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
        for filename in self.input.recurrences:
            recurrence = self.input.recurrences[filename]
            # recurrence = Recurrence(
            #     "eqs :=[s(n) = -4*s(n-2) + 4*s(n-1),s(0) = 6,s(1) = 8];")

    def print(self):
        """ """
        recurrences = self.input.recurrences
        for recurrence in recurrences:
            print("File: " + recurrence)
            recurrences[recurrence].print()
            print("")


if __name__ == '__main__':
    solver = Main()
    solver.run()
