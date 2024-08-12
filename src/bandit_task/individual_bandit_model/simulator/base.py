from abc import ABC, abstractmethod


class BaseSimulator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make_choice(self):
        pass

    def learn(self, choice, reward):
        pass
