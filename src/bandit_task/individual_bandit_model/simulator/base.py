from abc import ABC, abstractmethod


class BaseSimulator(ABC):
    def __init__(self):
        self.q_values = None

    @abstractmethod
    def make_choice(self):
        pass

    def learn(self, choice, reward):
        pass
