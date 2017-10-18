import re


class Solver:

    def __init__(self, recurrence):
        self.recurrence = recurrence
        self.homo = ""
        self.fn = ""

        # execute stack
        self.exe = []
        # atom stack
        self.atoms = []

    def solve(self):
        self.recurrence.print()
        print("****")
        print("")
        print(self.terms())
        print("")

    def terms(self):
        recurrence = self.recurrence.recurrence
        match = re.search("s\(n\)=(.+)", recurrence)

        if not match:
            raise SyntaxError("No recurrence found in \"" + recurrence + "\".")

        equation = match.group(1)

        terms = []
        atoms = []

        evaluating = equation
        while evaluating != "":
            atom = re.match("([a-zA-Z0-9]+)", evaluating)
            if atom:
                atoms.append(evaluating[atom.start(1):atom.end(1)])
                evaluating = evaluating[atom.end(1):]
            else:
                evaluating = evaluating[1:]

        evaluating = equation
        atom_index = 0
        level = 0
        term = ""
        prev = "OPERATOR"
        while evaluating != "":
            if atom_index < len(atoms) \
                    and evaluating.find(atoms[atom_index]) == 0:
                # Currently evaluating an atom, so we add it to our term.
                term = term + atoms[atom_index]

                # Now update the to evaluate part.
                evaluating = evaluating[len(atoms[atom_index]):]

                # Check for the next atom
                atom_index = atom_index + 1
                prev = "ATOM"
            elif evaluating[0] in "-+*/^":
                # Currently evaluating an operator
                op = evaluating[0]

                # We check for term dividers, which can only occur at the
                # root level.

                # TODO: Convert the case of a minus term to * -1
                if op in "+-" and level == 0:
                    # Now we check whether we are dealing with an unary or not.
                    # It is an unary if the previous state was an OPERATOR.
                    if prev == "ATOM" or prev == "GROUP":
                        # We found the end of a term.
                        terms.append(term)
                        term = ""
                else:
                    # Else just add the operator to the term.
                    term = term + op
                evaluating = evaluating[1:]
                prev = "OPERATOR"
            elif evaluating[0] in "()":
                group = evaluating[0]

                if group == "(":
                    level = level + 1
                elif group == ")":
                    level = level - 1
                term = term + group
                evaluating = evaluating[1:]
                prev = "GROUP"
            else:
                raise SyntaxError("Invalid syntax.")

        if term != "":
            terms.append(term)

        return terms

    def termss(self):
        recurrence = self.recurrence.recurrence
        match = re.search("s\(n\)=(.+)", recurrence)

        if not match:
            raise SyntaxError("No recurrence found in \"" + recurrence + "\".")

        equation = match.group(1)
        terms = []

        operators = re.findall("[-+*/()^]", equation)
        operators.reverse()
        # atoms = re.findall("([a-zA-Z0-9]+)[-+*/()^]", equation)
        atoms = self.parseaatoms(equation)

        term = ""
        level = 0

        for a in atoms:
            op = operators.pop()

            if a != "PLACEHOLDER":
                term = term + a

            if op in "+-" and level == 0:
                terms.append(term)
                term = ""
            else:
                term = term + op
                if op == "(":
                    level = level + 1
                elif op == ")":
                    level = level - 1
        if term != "":
            terms.append(term)

        return terms

    def parseatoms(self, equation):
        atoms = []

        isatom = re.match("([a-zA-Z0-9]+)", equation)
        isoperator = re.match("([-+*/()^])", equation)
        if isatom:
            atoms.append(equation[isatom.start(1):isatom.end(1)])
            atoms = atoms + self.parseatoms(equation[isatom.end(1):])
        elif isoperator:
            if len(equation) > 1:
                # Place a placeholder if another operator appears right after,
                # this would desync the lengths of the stacks
                if equation[1] in "(-":
                    # if this is the case, then add a placeholder atom.
                    atoms.append("PLACEHOLDER")
            atoms = atoms + self.parseatoms(equation[1:])

        return atoms

    def parseaatoms(self, equation):
        atoms = []

        prev = "OPERATOR"
        unary_operators = "-+"

        i = 0
        while len(equation[i:]) > 0:
            temp = equation[i:]
            isatom = re.match("([a-zA-Z0-9]+)", temp)
            isoperator = re.match("([-+*/^])", temp)

            if isatom:
                atoms.append(temp[isatom.start(1):isatom.end(1)])
                i = i + isatom.end(1)
                prev = "ATOM"
            elif isoperator:
                if prev == "OPERATOR":
                    if temp[0] not in unary_operators:
                        raise SyntaxError("Invalid syntax.")
                    atoms.append("UNARY")
                prev = "OPERATOR"
                i = i + 1

        return atoms
