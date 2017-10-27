class Menu:
    """Menu class."""

    def __init__(self):
        """"""

    def welcome(self):
        print("Welcome to the recurrence solver, made by Quoc An Ha.")
        print("")

    def input(self, recurrences):
        print("We've found the following files available for input.")

        print("\t0. All recurrences below.")

        i = 1
        for name in recurrences:
            recurrence = recurrences[name]

            print("\t{}. {}".format(i, name))
            print("\t\t s(n) = {}".format(recurrence.recurrence))

            i = i + 1

        sel = None

        while sel is None or sel > len(recurrences):
            sel = int(input("Make a selection: "))

            if sel < 0 or sel > len(recurrences):
                print("Invalid selection, please try again.")

        print()

        return sel
