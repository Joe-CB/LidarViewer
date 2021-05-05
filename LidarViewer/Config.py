import os
import json
from scipy import signal
import math


class Config:
    def __init__(self, filename: str='cache.json'):
        self.config = {}
        self.filename = filename
        self.load()

    def addConfig(self, key, value):
        self.config[key] = value
        self.save()

    def getConfig(self, key):
        return self.config.get(key)

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.config, f)
    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                self.config = json.load(f)