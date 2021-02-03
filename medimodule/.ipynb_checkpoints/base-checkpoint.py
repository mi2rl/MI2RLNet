from abc import *

class BaseModule(metaclass=ABCMeta):
    @abstractmethod
    def init(self, weight_path):
        pass

    @abstractmethod
    def _preprocessing(self, path):
        pass

    @abstractmethod 
    def predict(self, img):
        pass
        
