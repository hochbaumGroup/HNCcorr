from copy import deepcopy


class Config:
    def __init__(self, **entries):
        self._entries = entries

        for key, value in self._entries.items():
            setattr(self, key, value)

    def __add__(self, other):
        entries = deepcopy(self._entries)
        entries.update(other._entries)  # pylint: disable=W0212

        return Config(**entries)
