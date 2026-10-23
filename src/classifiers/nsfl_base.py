from abc import ABC, abstractmethod
import numpy.typing as npt

# Base encoder interface, has a name          
                          
class IEncoder(ABC):
    name: str = "base_encoder"
                
class ITextEncoder(IEncoder):
    @abstractmethod
    def encode_texts(self, texts) -> npt.ArrayLike:
        raise NotImplementedError
    
class IImageEncoder(IEncoder):
    @abstractmethod
    def encode_images(self, images) -> npt.ArrayLike:
        raise NotImplementedError

# New: combined interface requiring both text and image encoding
class IImageTextEncoder(IImageEncoder, ITextEncoder):
    """Encoder that supports both image and text encoding."""
    pass
