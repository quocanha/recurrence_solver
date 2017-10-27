from sympy import *
import re

from sympy.core.numbers import NegativeOne, One


class Recurrence:

    filename = ""
    raw = ""

    recurrence = None
    homo = []
    nonhomo = []
    initials = {}

    solution = None

    def __init__(self, raw, filename=""):
        self.filename = filename
        self.raw = raw

        self.recurrence = None
        self.initials = {}
        self.homo = []
        self.nonhomo = []
        self.parse()
        self.solution = None

    def parse(self):
        """
            Parses the raw equation into it's recurrence equation and initial
            conditions.
        """
        matches = re.match("eqs :=\[(.+?)]", self.raw)
        if matches:
            equations = matches.group(1).replace(" ", "")
            equations = re.findall(
                "s\(([n0-9])\)=([a-zA-Z0-9*/+\-^=()]+)", equations)

            for inp, equation in equations:
                if re.match("[a-zA-Z]", inp):
                    self.recurrence = sympify(equation)
                else:
                    self.initials[inp] = equation
        else:
            raise SyntaxError("Invalid equation.\n {}".format(self.raw))

        self.parse_homogeneous()

    def solve(self):
        print("Solving...")

        characteristic, facts, k = self.find_roots()
        print("\tCharacteristic equation:")
        print("\t\t%s" % str(characteristic))
        print("\tRoots:")
        print("\t\t%s" % str(facts))

        homogeneous = self.homo_sequence(facts, k)
        print("\tHomogeneous solution:")
        print("\t\t{}".format(homogeneous))

        # self.test_homo(homo_solution)

        if len(self.nonhomo) == 0:
            print("\tNo inhomogeneous part.")
            alphas = self.solve_alphas(homogeneous)
            solution = self.sub_alphas(homogeneous, alphas)
            print(solution)
            self.solution = solution
            return solution
        else:
            print("\tFound inhomogeneous part.")

            solved = False
            solution = None

            print("\tTrying to solve while splitting all fn's.")
            fn_s = self.get_linear_parts_fn(False, True)

            try:
                solution = self.solve_theorem6(fn_s, facts)
            except ArithmeticError:
                print("\t\tFailed to apply theorem 6.")
                pass
            else:
                solved = True

            if not solved:
                try:
                    solution = self.solve_guesss(fn_s)
                except ArithmeticError:
                    print("\t\tFailed to guess.")
                    pass
                else:
                    solved = True

            if solved and solution is not None:
                solution = homogeneous + solution
                alphas = self.solve_alphas(solution)
                solution = self.sub_alphas(solution, alphas)

                print("\tGeneral nonhomogeneous solution:")
                print("\t\t{}".format(solution))

                self.solution = solution
                return solution
            raise ArithmeticError("No solution found.")

    def solve_guesss(self, fn_s):
        s = Function("s")
        n = symbols("n")

        homo = s(n) - Add(*self.homo)
        particulars = []

        for fn in fn_s:
            form, p_count = self.guess_exponential(homo, fn)
            if form is None:
                form, p_count = self.guess_polynomial(homo, fn)
            if form is None:
                # Now what lol.
                raise ArithmeticError("No solution found.")
            else:
                if type(form) == Rational:
                    particulars.append(form)
                else:
                    recurrence = homo - fn
                    particular = self.insert_particular_solution(
                        recurrence, form, p_count)
                    particulars.append(particular)

        solution = S.Zero
        for particular in particulars:
            solution = solution + particular
        return solution

    def guess_exponential(self, homogeneous, fn):
        s = Function("s")
        n = symbols("n")

        p0, p1, p2 = symbols("p0 p1 p2")
        exponential = p2 * p1**n + p0
        g = lambdify(n, exponential)
        a = Wild("a")
        eq = homogeneous.replace(s(a), g(a))

        result = solve(Eq(eq, fn), (p0, p1, p2))
        if type(result) == dict:
            p_count = 3
            solution = exponential

            i = 0
            while i <= p_count:
                p = symbols("p{}".format(i))
                if p in result:
                    solution = solution.replace(p, result[p])
                else:
                    solution = solution.replace(p, 0)
                i = i + 1

            return solution, p_count
        else:
            return None, 0

    def guess_polynomial(self, homogeneous, fn):
        s = Function("s")
        n = symbols("n")

        degree = 1
        result = None
        guess = None
        solved = False
        while degree < 10 and solved is False:
            print("\tGuessing with degree {}...".format(degree))
            guess = self.create_polynomial_function(degree)

            g = lambdify(n, guess)
            a = Wild("a")
            eq = homogeneous.replace(s(a), g(a))

            p_str = ""
            i = 0
            while i <= degree:
                p_str = p_str + "p{} ".format(i)
                i = i + 1

            p = symbols(p_str)

            result = solve(Eq(eq, fn), p)

            if type(result) == dict:
                print("\tPolynomial guess!")
                solved = True
            degree = degree + 1
        if solved:
            p_count = degree - 1
            solution = guess

            i = 0
            while i <= p_count:
                p = symbols("p{}".format(i))
                if p in result:
                    solution = solution.replace(p, result[p])
                else:
                    solution = solution.replace(p, 0)
                i = i + 1

            return solution, p_count
        else:
            return None, None, 0

    def solve_theorem6(self, fn_s, facts):
        s = Function("s")
        n = symbols("n")

        particulars = []

        for fn in fn_s:
            form, p_count = self.get_particular_form(fn, facts)

            recurrence = s(n) - Add(*self.homo, fn)
            if form is not None:
                solution = self.insert_particular_solution(
                    recurrence, form, p_count)

                if solution is not None:
                    particulars.append(solution)

        solution = S.Zero
        for particular in particulars:
            solution = solution + particular
        return solution

    def insert_particular_solution(self, recurrence, form, p_count):
        a = Wild("a")
        s = Function("s")
        n = symbols("n")

        # Lambdify the form so we can substitute it.
        particular = lambdify(n, form)
        # Substitute the form into the recurrence
        inserted = recurrence.replace(s(a), particular(a))
        inserted = factor(inserted)

        # Now build the variable symbols
        i = 0
        pees_str = ""
        while i < p_count:
            pees_str = pees_str + "p" + str(i) + " "
            i = i + 1
        pees_s = symbols(pees_str)

        # Now try to solve the recurrence with our form substituted in,
        # For every variable in the form except n.
        if type(pees_s) == Symbol:
            result = solve(inserted, pees_s)
        else:
            result = solve(inserted, *pees_s)

        if type(result) == dict:
            pees = result
            solution = form

            for pee in pees:
                solution = solution.replace(pee, pees[pee])

            # Woo, it worked
            return simplify(solution)
        elif type(result) == list:
            solution = form

            i = 0
            while i < p_count:
                p = symbols("p{}".format(i))
                solution = solution.replace(p, result[i])
                i = i + 1
            print("Caution: assumed list type, with result[0] is a correct result. You should check if there is no n variable in this.")
            return simplify(solution)
        else:
            raise ArithmeticError("No solution found.")

    def get_linear_parts_fn(self, ask, split):
        terms = self.nonhomo[:]
        parts = []

        if ask:
            while len(terms) > 0:
                first = terms.pop()
                print("\tAssembling Fn part 1 ({}):".format(first))

                part_list = []
                for term in terms:
                    yes = input("\t\tIs {} part of this group? [Y/n]:".format(term))
                    if yes in ["", "y", "yes", "YES"]:
                        part_list.append(term)

                part = first
                for p in part_list:
                    terms.pop(terms.index(p))
                    part = part + p

                parts.append(part)
        elif not split:
            parts = [Add(*self.nonhomo)]
        else:
            parts = self.nonhomo[:]
        return parts

    def get_particular_form(self, fn, facts):
        # Assuming each part is a polynomial.
        n = symbols("n")
        p_count = 0

        if type(fn) == Integer:
            solution, p_count = self.b_integer(fn, p_count, facts)
            return solution, p_count
        else:
            try:
                b = Poly(fn)
            except Exception as e:
                # It's not a polynomial, use s = 1
                # solution, p_count = self.b_poly(
                #     1, solution, part, p_count, facts)
                pass
                solution = None
                p_count = 0
                return solution, p_count
            else:
                a = Wild("a")
                if b.gen == n:
                    # We have a polynomial with generator n.
                    solution, p_count = self.b_poly(
                        1, Poly(b, 1**n), p_count, facts)
                    return solution, p_count
                elif type(b.gen) == Pow and b.gen.args[1] == n:
                    # We have a polynomial with generator s^n.
                    s = b.gen.args[0]
                    solution, p_count = self.b_poly(
                        s, fn, p_count, facts)
                    return solution, p_count
                else:
                    # ahh fuck it, try this.
                    raise Exception("Unresolved case.")

    def create_polynomial_function(self, degree):
        n = symbols("n")
        p = symbols("p0")

        fn = p
        i = 1
        while i <= degree:
            p = symbols("p{}".format(i))
            fn = fn + p*n**i
            i = i + 1
        return fn

    def is_root(self, s, facts):
        r = symbols("r")
        # Now check if s is a root.
        for fact, mp in facts:
            result = solve(Eq(fact, 0), r)[0]
            if result == s:
                return mp
        return None

    def b_integer(self, part, p_count, facts):
        n = symbols("n")
        if type(part) == Integer:
            # p0 * (1)^n == p0 if the term is an integer.
            p = symbols("p{}".format(p_count))
            s = 1

            trm = p
            # Check if s is a root, if yes then apply n^m to
            # whole function.
            m = self.is_root(s, facts)
            if m is not None:
                trm = n**m * trm
            p_count = p_count + 1
            solution = trm
            return solution, p_count

    def b_poly(self, s, part, p_count, facts):
        n = symbols("n")
        poly = Poly(part)

        terms = poly.terms()
        t = []
        for exp, term in terms:
            t.append(term)

        b = Poly(*t, n)

        p = symbols("p{}".format(p_count))
        p_count = p_count + 1
        trm = p

        monomial, coeff = b.LT()
        t = monomial.exponents[0]

        i = t
        while i > 0:
            p = symbols("p{}".format(p_count))
            trm = trm + p * n ** i
            p_count = p_count + 1
            i = i - 1

        # Check if s is a root, if yes then apply n^m to
        # whole function.
        m = self.is_root(s, facts)
        if m is not None:
            trm = n**m * trm

        solution = trm * s**n
        return solution, p_count

    def general_nonhomo_sequence(self, facts, ask=False, split=True):
        if len(self.nonhomo) <= 0:
            return S.Zero, 0

        n, r = symbols("n, r")

        p_count = 0
        solution = S.Zero

        terms = self.nonhomo[:]
        parts = []

        if ask:
            while len(terms) > 0:
                first = terms.pop()
                print("\tAssembling Fn part 1 ({}):".format(first))

                part_list = []
                for term in terms:
                    yes = input("\t\tIs {} part of this group? [Y/n]:".format(term))
                    if yes in ["", "y", "yes", "YES"]:
                        part_list.append(term)

                part = first
                for p in part_list:
                    terms.pop(terms.index(p))
                    part = part + p

                parts.append(part)
        elif not split:
            parts = [Add(*self.nonhomo)]
        else:
            parts = self.nonhomo[:]

        for part in parts:
            # Assuming each part is a polynomial.

            if type(part) == Integer:
                solution, p_count = self.b_integer(
                    part, p_count, facts)
            else:
                try:
                    b = Poly(part)
                except Exception as e:
                    # It's not a polynomial, use s = 1
                    # solution, p_count = self.b_poly(
                    #     1, solution, part, p_count, facts)
                    pass
                    solution = None
                    p_count = 0
                else:
                    a = Wild("a")
                    if b.gen == n:
                        # We have a polynomial with generator n.
                        solution, p_count = self.b_poly(
                            1, poly, p_count, facts)
                    elif type(b.gen) == Pow and b.gen.args[1] == n:
                        # We have a polynomial with generator s^n.
                        s = b.gen.args[0]
                        solution, p_count = self.b_poly(
                            s, part, p_count, facts)
                    else:
                        # ahh fuck it, try this.
                        raise Exception("Unresolved case.")

        return solution, p_count

        #     elif type(part) == Pow:
        #         if type(part.args[1]) == Add:
        #             if part.args[1].args[1] == n:
        #                 # If the power is an instance of coeff ^ (n - i),
        #                 # that means we can split it into coeff ^ -i * coeff ^ n
        #                 # so s then is coeff, with one in the polynomial.
        #                 coeff = part.args[0]
        #                 s = coeff
        #         elif part.args[1] == n:
        #             s = part.args[0]
        #
        #         p = symbols("p{}".format(p_count))
        #         p_count = p_count + 1
        #         trm = p * s ** n
        #
        #         # Check if s is a root, if yes then apply n^m to
        #         # whole function.
        #         m = self.is_root(s, facts)
        #         if m is not None:
        #             trm = n**m * trm
        #         solution = solution + trm


        # fn = Add(*self.nonhomo)
        #
        # s = Pow(1, n)
        # s_coeff = 1
        #
        # # Find s by seeing which term is a power.
        # for arg in fn.args:
        #     if type(arg) == Pow:
        #         pow = arg
        #         for pow_arg in pow.args:
        #             if type(pow_arg) == Add:
        #                 s = pow
        #                 s_coeff = pow_arg.args[0]
        #
        # # The polynomial is Fn without the power term.
        # b = Poly(fn / s)
        #
        # # Now get the term with the highest power, call it t.
        # monomial, coeff = b.LT()
        # t = monomial.exponents[0]
        #
        # # Create the general particular solution based on t.
        # part = S.Zero
        # i = t
        # while i >= 0:
        #     p = symbols("p{}".format(i))
        #     part = part + p * n ** i
        #     i = i - 1
        #
        # solution = part * s
        #
        # # Now check if s is a root.
        # for fact, mp in facts:
        #     rt = solve(Eq(fact, 0), r)[0]
        #     if rt == s_coeff:
        #         # If it is a root, then multiply by n^m
        #         solution = n**mp * solution
        #         break
        #
        # return expand_mul(solution), t

    def test_homo(self, solution):
        n = symbols("n")
        s = Function("s")
        f = s(n) - Add(*[x for x in self.homo])
        inits = {s(int(key)): int(value) for (key, value) in self.initials.items()}

        try:
            result = rsolve(f, s(n), inits)
        except Exception as e:
            print("\tRsolve exception:")
            print("\t\t{}".format(e))
        else:
            print("Rsolve:")
            print("\t\t{}".format(result))

            if result is not None:
                test = simplify(solution - result) == 0
                if test:
                    print("Test: {}".format(test))
                else:
                    i = 0
                    print("\t\tSolution\tRSolve:")
                    while i < 20:
                        s = simplify(solution.subs(n, i))
                        r = simplify(result.subs(n, i))
                        print("\t\t{}\t\t\t{}".format(s, r))
                        i = i + 1

    def sub_alphas(self, sequence, alphas):
        solution = sequence

        # Substitute every alpha into the sequence.
        for a in alphas:
            alpha = symbols("a{}".format(a))
            solution = solution.subs(alpha, alphas[a])

        solution = simplify(solution)

        return solution

    def solve_alphas(self, equation):
        alphas = {}
        i = 0
        for key in self.initials:
            variable = symbols("a{}".format(i))
            alpha = self.solve_initial(alphas, equation, key, variable)
            alphas[i] = alpha
            i = i + 1

        return alphas

    def solve_initial(self, alphas, general, key, var):
        n = symbols("n")
        initial = self.initials[key]
        inp = sympify(int(key))
        out = sympify(initial)

        general = general.subs(n, inp)

        for a in alphas:
            alpha = symbols("a{}".format(a))
            general = general.subs(alpha, alphas[a])

        eq = Eq(general - out)
        result = solve(eq, var)
        return result[0]

    def homo_sequence(self, facts, k):
        n = symbols("n")
        distinct = []

        # Build the alpha's for each distinct root, of multiplicity m
        a_count = 0
        for root, mp in facts:
            r = symbols("r")
            root = solve(Eq(root, 0), r)[0]
            eq = root ** n

            a = symbols("a{}".format(a_count))
            a_count = a_count + 1
            i = 1

            # We have mp amount of alpha's for this root.
            while i < mp:
                new_a = symbols("a{}".format(a_count))
                b = n ** (i - 1)
                c = n ** b
                new_a = new_a * c
                a = a + new_a
                a_count = a_count + 1
                i = i + 1

            eq = eq * a
            distinct.append(eq)

        if a_count != k:
            raise Exception(
                "The total amount of roots does not match. Expected {}, got {}."
                .format(k, a_count)
            )

        general = simplify(Add(*distinct))
        return Add(*distinct)

    def get_characteristic(self, recurrence):
        """
        Returns the characteristic equations of a recurrence
        :param recurrence:
        :return:
        """
        # Define the symbols we're using for sympy.
        r, s, n= symbols("r s n")
        # A wildcard symbol for the replacement.
        a = Wild("a")

        # Keep track of the smallest term of the homogeneous part.
        smallest_term = n

        for term in recurrence.args:
            inp = self.find_function_input(term)
            if solve(inp < smallest_term):
                smallest_term = inp

        return simplify(recurrence.replace(s(a), r**a) / r ** smallest_term)

    def find_roots(self):
        # Define the symbols we're using for sympy.
        n, r, s = symbols("n r s")
        # A wildcard symbol for the replacement.
        a = Wild("a")

        # Keep track of the smallest term of the homogeneous part.
        smallest_term = n

        # Rebuild the homogeneous part of the equation by adding all the
        # linear terms.
        terms = []
        for term in self.homo:
            # Store the smallest term so we can simplify the equation by it.
            inp = self.find_function_input(term)
            if solve(inp < smallest_term):
                smallest_term = inp

            # Append each term to the new equation.
            terms.append(term.replace(s(a), r**a))

        # Create the right hand side of the equation by adding all the terms.
        rhs = Add(*terms) / r**smallest_term

        # The left hand side is r^n divided by the smallest term, since
        # these can be factored out.
        lhs = r**n / r**smallest_term

        # Create the equation by subtracting the rhs from the lhs.
        characteristic = simplify(lhs - rhs)

        # Now find the characteristic roots.
        (coeff, factors) = factor_list(characteristic)
        facts = self.get_multiplicity(factors, characteristic)

        # The degree of the characteristic polynomial.
        k = 0
        for arg in smallest_term.args:
            if type(arg) == Integer:
                k = arg.p * -1
            elif type(arg) == NegativeOne:
                k = 1
            elif type(arg) == One:
                k = -1

        if sum([mp for (fact, mp) in facts]) != k:
            print("\tTheoretical approach failed, bruteforcing multiplicities...")
            roots = solve(characteristic, r)
            facts = self.bruteforce_multiplicity(characteristic, roots)

        return characteristic, facts, k

    def get_multiplicity(self, factors, characteristic):
        """
        Finds the multiplicity of a factor using the results of factor_list.

        :param factors:
        :param characteristic:
        :return:
        """
        roots = []

        for fact, mp in factors:
            # We know the factor has at least a multiplicity of 1
            occurrences = mp

            # Check if there are more occurrences of the factor in the
            # characteristic equation.
            more = true
            while more:
                # Now we divide the characteristic equation by the factor
                simplified = simplify(characteristic / fact ** mp)

                # If the factor still is a factor in the divided equation
                # then there are more occurrences of the factor in the equation.
                (s_coeff, s_factors) = factor_list(simplified)
                if fact in [f for (f, multip) in s_factors]:
                    occurrences = occurrences + 1
                else:
                    more = false

            roots.append((fact, occurrences))

        return roots

    def bruteforce_multiplicity(self, eq, roots):
        """
        Bruteforces the multiplicity of the roots in an equation by dividing
        and checking if it's still a polynomial afterwards.
        :param eq:
        :param roots:
        :return:
        """
        result = []
        for root in roots:
            r = symbols("r")

            simplified = eq
            fact = (r - root)

            occurences = 0
            i = 0
            j = 1
            while i < 20:
                previous = simplified
                simplified = simplify(previous / fact)

                if type(simplified) == Add:
                    # Still a polynomial.
                    occurences = occurences + j
                    j = 0
                elif simplified == S.One:
                    occurences = occurences + j
                    j = 0
                i = i + 1
                j = j + 1

                # if i % 10 == 0:
                #     print(i)

            result.append((fact, occurences))
        return result

    def find_function_input(self, term):
        if term.is_Function:
            return term.args[0]
        elif len(term.args) > 0:
            result = None
            for arg in term.args:
                inp = self.find_function_input(arg)
                if inp is not None:
                    result = inp
            return result
        return None

    def parse_homogeneous(self):
        terms = self.terms()

        for term in terms:
            if self.is_homogeneous(term):
                self.homo.append(term)
            else:
                self.nonhomo.append(term)

        return terms

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
        """
        Retrieves the terms of the recurrence.
        :return:
        """
        terms = []

        if type(self.recurrence) is Add:
            # We have multiple terms
            terms = self.recurrence.args
        else:
            # We do not have multiple terms, so the whole expression is one term
            terms.append(self.recurrence)

        return terms

    def __str__(self):
        rec = "Raw:\n\t{}\nRecurrence:\n\ts(n) = {}\n"
        rec = rec.format(self.raw, self.recurrence)

        rec = rec + "Initials:\n"
        for initial in self.initials:
            rec = rec + "\t- s({}) = {}".format(initial, self.initials[initial]) \
                  + "\n"

        rec = rec + "Homogeneous:\n"
        rec = rec + "\t{}".format(self.homo)
        rec = rec + "\n"

        rec = rec + "Nonhomogeneous:\n"
        rec = rec + "\t{}".format(self.nonhomo)

        return rec
