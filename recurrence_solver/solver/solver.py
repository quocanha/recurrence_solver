from sympy import *
from recurrence.recurrence import Recurrence


class Solver:

    recurrence = None

    def __init__(self, recurrence):
        if type(recurrence) is not Recurrence:
            raise TypeError("Solver requires a recurrence of type Recurrence")

        self.recurrence = recurrence

    def solve(self):
        print("**** Recurrence ****")
        print(self.recurrence)
        terms = self.terms()

        homo = []
        non = []
        for term in terms:
            if self.is_homogeneous(term):
                homo.append(term)
            else:
                non.append(term)

        print("Homogeneous terms:")
        for term in homo:
            print("\t- " + str(term))
        print("Nonhomogeneous terms:")
        for term in non:
            print("\t- " + str(term))
        print()

    def is_homogeneous(self, term):
        """
        Traverses a term until it finds a function as argument.
        :param term:
        :return:
        """
        if term.is_Function:
            return true
        else:
            is_homo = false
            for arg in term.args:
                is_homo = self.is_homogeneous(arg)
            return is_homo

    def terms(self):
        terms = []
        expression = sympify(self.recurrence.recurrence)

        var = expression[list(expression.keys())[0]]
        equation = expression[list(expression.keys())[1]]

        if type(var) is not Symbol or not var.is_Atom or not var.is_symbol:
            raise TypeError(
                "Expected the first element of the expression to be a variable."
            )

        if type(equation) is Add:
            # We have multiple terms
            terms = equation.args
        else:
            # We do not have multiple terms, so the whole expression is one term
            terms.append(expression)

        return terms
