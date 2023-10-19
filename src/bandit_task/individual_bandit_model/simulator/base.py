from abc import ABC, abstractmethod


class BaseSimulator(ABC):
    @abstractmethod
    def make_choice(self):
        pass

    def learn(self, choice, reward):
        pass
