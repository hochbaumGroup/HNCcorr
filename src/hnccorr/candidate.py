class Candidate:
    def __init__(self, value):
        self._value = value

    def __eq__(self, other):
        return self._value == other._value
