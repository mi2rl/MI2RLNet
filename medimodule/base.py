from abc import *

class BaseModule(metaclass=ABCMeta):
    @abstractmethod
    def _preprocessing(self, path):
        pass

    @abstractmethod 
    def predict(self, img):
        pass
        
