import re


class Recurrence:

    def __init__(self, raw):
        self.raw = raw

        self.equations = []
        self.recurrence = ""
        self.initials = []

        self.parse(self.raw)

    def parse(self, raw):
        """
            Parses the raw equation into it's recurrence equation and initial
            conditions.
        """
        matches = re.match("eqs :=\[(.+?)\]", raw)
        if matches:
            eqs = matches.group(1).replace(" ", "")
            self.equations = re.findall(
                "s\([n0-9]\)=[a-zA-Z0-9\*\+\-\^\=\(\)]+", eqs)

            self.recurrence = self.equations[0]
            self.initials = self.equations[1:]

    def print(self):
        print("Raw: " + self.raw)
        print("Recurrence: " + self.recurrence)
        print("Initial conditions:")
        for initial in self.initials:
            print("- " + initial)
