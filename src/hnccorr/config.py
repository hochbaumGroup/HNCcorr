class Config:
    def __init__(self, **entries):
        self.entries = entries

        for key, value in entries.items():
            setattr(self, key, value)
