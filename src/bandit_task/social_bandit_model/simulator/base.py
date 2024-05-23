from abc import ABC, abstractmethod


class BaseSimulator(ABC):
    @abstractmethod
    def make_choice(self):
        pass

    @abstractmethod
    def learn_from_own(self, choice: int, reward: float) -> None:
        pass

    @abstractmethod
    def learn_from_partner(self, choice: int, reward: float) -> None:
        pass
