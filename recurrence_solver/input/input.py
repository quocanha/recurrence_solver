from os import listdir
from os.path import isfile, join

from recurrence.recurrence import Recurrence


class Input:
    """Input class that reads a certain directory for files and reads them."""

    def __init__(self, path):
        self.path = path
        self.recurrences = {}

        self.readfiles()

    def readfiles(self):
        files = [f for f in listdir(self.path) if isfile(join(self.path, f))]

        for file in files:
            temp = open(self.path + "/" + file)
            raw = ""

            for line in temp.readlines():
                raw += line.rstrip()

            self.recurrences[file] = Recurrence(raw)


