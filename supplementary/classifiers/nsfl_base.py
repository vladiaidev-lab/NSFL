import os
import torch
import functools
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TypeVar, Generic
from jaxtyping import Array, Bool
from pylogics.syntax.base import Formula
from pylogics.parsers import parse_pl

def flatten(l):
    return [item for sublist in l for item in sublist]
    

PredMatrix = Bool[Array, "h w"]  # shape: (n_documents, n_labels)

class Document:
    """Base document type - concrete docs should inherit this."""
    pass

class Query:
    """Base query type - concrete queries should inherit this."""
    pass

T = TypeVar("T", bound=Document)
Q = TypeVar("Q", bound=Query)

class TextDocument(Document):
    text: str
    
    def __init__(self, text: str):
        self.text = text

class ImageDocument(Document):
    image_path: str
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        
class TextQuery(Query):
    query_text: str
    
    def __init__(self, query_text: str):
        self.query_text = query_text
        
class PropositionalQuery(Query):
    query_formula: Formula
    
    def __init__(self, query_text: str):
        self.query_formula = parse_pl(query_text)
        
    def get_atom_names(self):
        atoms = []
        
        def dfs(node):
            # Check if it's a leaf node (Atomic)
            if node.__class__.__name__ == "Atomic":
                atoms.append(node.name)
            # Check for multiple operands
            elif hasattr(node, "_operands") and node._operands:
                for operand in node._operands:
                    dfs(operand)
            # Check for single operand
            elif hasattr(node, "_arg"):
                dfs(node._arg)
        
        dfs(self.query_formula)
        return [x.replace('_', ' ') for x in np.unique(atoms).tolist()]     # This defines order
        
    
          
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
    
class IMultiLabelClassificationDataset(ABC, Generic[Q,T]):
    name: str = "base_dataset"
    """Abstract interface for multi-label classification datasets parametrized by query and document type."""

    @abstractmethod
    def get_gt_matrix(self) -> PredMatrix:
        """Return ground-truth label matrix (n_documents x n_labels).

        Each row corresponds to a document and each column to a label.
        The matrix can be binary or contain scores.
        """
        raise NotImplementedError

    @abstractmethod
    def get_documents(self) -> List[T]:
        """Return the list of document identifiers or texts."""
        raise NotImplementedError

    @abstractmethod
    def get_queries(self) -> List[Q]:
        """Return the list of queries (if applicable) for this dataset."""
        raise NotImplementedError
    
    @abstractmethod
    def get_gt_dataframe(self) -> pd.DataFrame:
        """Return the dataset as a pandas DataFrame."""
        raise NotImplementedError

class MLCDatasetBase(IMultiLabelClassificationDataset[Q,T], Generic[Q,T]):
    """
        Base class for multi-label classification datasets.
        It assumes _gt_matrix, _documents, _queries are defined in subclass init
    """
    
    pruned: bool = False
    valid_documents: List[int] = field(default_factory=list)
    valid_queries: List[int] = field(default_factory=list)
    
    def prune_gt_matrix(
        self,
        minimal_query_hits: int = 1,
        minimal_image_hits: int = 1
    ) -> None:
        """Prune the ground truth matrix to remove queries and images with minimal hits."""
        from experiments.fuzzy_logic.classifiers.nsfl_utils import stable_prune_gt_matrix
        
        pruned_matrix, valid_documents, valid_queries = stable_prune_gt_matrix(
            self.get_gt_matrix(),
            minimal_query_hits,
            minimal_image_hits
        )
        
        # Update internal state
        self._gt_matrix = pruned_matrix
        self._documents = [self._documents[i] for i in valid_documents]
        self._queries = [self._queries[i] for i in valid_queries]
        self.pruned = True
        self.valid_documents = valid_documents
        self.valid_queries = valid_queries


    def get_gt_matrix(self) -> PredMatrix: # type: ignore
        return self._gt_matrix

    def get_documents(self) -> List[T]:
        return self._documents
        
    def get_queries(self) -> List[Q]:
        return self._queries
    
class IDatasetEncoder(ABC):
    @abstractmethod
    def encode_queries(self) -> npt.ArrayLike:
        raise NotImplementedError
    @abstractmethod
    def encode_documents(self) -> npt.ArrayLike:
        raise NotImplementedError
    @abstractmethod
    def get_dataset(self) -> IMultiLabelClassificationDataset:
        raise NotImplementedError
    @abstractmethod
    def get_encoder(self) -> IEncoder:
        raise NotImplementedError

class IStrategy(ABC):
    def strategy_keys(self) -> List[str]:
        """Return the list of keys produced by this strategy."""
        return ["default"]

# New strategy interfaces: separate concerns for query/doc encoding
class IQueryEncoderStrategy(IStrategy, Generic[Q]):
    """Strategy for encoding queries independently from documents."""
    @abstractmethod
    def encode(self, encoder: IEncoder, queries: List[Q]) -> Dict[str, npt.ArrayLike]:
        raise NotImplementedError
    

class IDocumentEncoderStrategy(IStrategy, Generic[T]):
    """Strategy for encoding documents independently from queries."""
    @abstractmethod
    def encode(self, encoder: IEncoder, documents: List[T]) -> Dict[str, npt.ArrayLike]:
        raise NotImplementedError

# Simple concrete strategies you can extend/replace
class PrefixTextQueryEncodingStrategy(IQueryEncoderStrategy[TextQuery]):
    
    def __init__(self, prefix: str):
        self.prefix = prefix
    """Encode TextQuery objects using an ITextEncoder-like encoder."""
    def encode(self, encoder: IEncoder, queries: List[TextQuery]) -> Dict[str, npt.ArrayLike]:
        if not hasattr(encoder, "encode_texts"):
            raise TypeError("Encoder does not implement text encoding interface")
        texts = [f"{self.prefix} {q.query_text}" for q in queries]
        # texts = [q.query_text for q in queries]
        embeddings = encoder.encode_texts(texts)
        return {"default": embeddings}


# Simple concrete strategies you can extend/replace
class TextQueryEncodingStrategy(IQueryEncoderStrategy[TextQuery]):
    """Encode TextQuery objects using an ITextEncoder-like encoder."""
    def encode(self, encoder: IEncoder, queries: List[TextQuery]) -> Dict[str, npt.ArrayLike]:
        if not hasattr(encoder, "encode_texts"):
            raise TypeError("Encoder does not implement text encoding interface")
        texts = [q.query_text for q in queries]
        embeddings = encoder.encode_texts(texts)
        return {"default": embeddings}
    
# Simple concrete strategies you can extend/replace
class PropositionalQueryEncodingStrategy(IQueryEncoderStrategy[TextQuery]):
    """Encode PropositionalQuery objects using an ITextEncoder-like encoder."""
    # We just encode the unique atom names
    def encode(self, encoder: IEncoder, queries: List[TextQuery]) -> Dict[str, npt.ArrayLike]:
        if not hasattr(encoder, "encode_texts"):
            raise TypeError("Encoder does not implement text encoding interface")
        texts = np.unique(flatten([q.get_atom_names() for q in queries])).tolist()      # This line implies an order (np.unique . flatten), and is repeated in pairs
        embeddings = encoder.encode_texts(texts)
        return {"default": embeddings}

class TextDocumentEncodingStrategy(IDocumentEncoderStrategy[TextDocument]):
    """Encode TextDocument objects using an ITextEncoder-like encoder."""
    def encode(self, encoder: IEncoder, documents: List[TextDocument]) -> Dict[str, npt.ArrayLike]:
        if not hasattr(encoder, "encode_texts"):
            raise TypeError("Encoder does not implement text encoding interface")
        texts = [d.text for d in documents] 
        embeddings = encoder.encode_texts(texts)
        return {"default": embeddings}

class ImageDocumentEncodingStrategy(IDocumentEncoderStrategy[ImageDocument]):
    """Encode ImageDocument objects using an IImageEncoder-like encoder."""
    def encode(self, encoder: IEncoder, documents: List[ImageDocument]) -> Dict[str, npt.ArrayLike]:
        if not hasattr(encoder, "encode_images"):
            raise TypeError("Encoder does not implement image encoding interface")
        paths = [d.image_path for d in documents]
        embeddings = encoder.encode_images(paths)
        return {"default": embeddings}

# Container strategies that combine multiple strategies
class CompositeQueryEncoderStrategy(IQueryEncoderStrategy[Q]):
    """Composite strategy that applies multiple query encoding strategies and unions their results."""
    def __init__(self, strategies: List[IQueryEncoderStrategy[Q]]):
        self.strategies = strategies
        
    def strategy_keys(self) -> List[str]:
        return list(set(key for strategy in self.strategies for key in strategy.strategy_keys()))
    
    def encode(self, encoder: IEncoder, queries: List[Q]) -> Dict[str, npt.ArrayLike]:
        result = {}
        for strategy in self.strategies:
            strategy_result = strategy.encode(encoder, queries)
            result.update(strategy_result)
        return result

class CompositeDocumentEncoderStrategy(IDocumentEncoderStrategy[T]):
    """Composite strategy that applies multiple document encoding strategies and unions their results."""
    def __init__(self, strategies: List[IDocumentEncoderStrategy[T]]):
        self.strategies = strategies
        
    def strategy_keys(self) -> List[str]:
        return list(set(key for strategy in self.strategies for key in strategy.strategy_keys()))
            
    def encode(self, encoder: IEncoder, documents: List[T]) -> Dict[str, npt.ArrayLike]:
        result = {}
        for strategy in self.strategies:
            strategy_result = strategy.encode(encoder, documents)
            result.update(strategy_result)
        return result

class NumpyQueryStrategyDecorator(IQueryEncoderStrategy[Q]):
    """Decorator for IQueryEncoderStrategy that converts torch.Tensor embeddings to numpy arrays."""
    def __init__(self, wrapped: IQueryEncoderStrategy[Q]):
        self.wrapped = wrapped

    def strategy_keys(self) -> List[str]:
        return self.wrapped.strategy_keys()

    def encode(self, encoder: IEncoder, queries: List[Q]) -> Dict[str, npt.ArrayLike]:
        raw = self.wrapped.encode(encoder, queries)
        converted: dict[str, npt.ArrayLike] = {}
        for key, emb in raw.items():
            if isinstance(emb, torch.Tensor):
                converted[key] = emb.detach().cpu().numpy()
            else:
                converted[key] = emb
        return converted

class NumpyDocumentStrategyDecorator(IQueryEncoderStrategy[T]):
    """Decorator for IQueryEncoderStrategy that converts torch.Tensor embeddings to numpy arrays."""
    def __init__(self, wrapped: IDocumentEncoderStrategy[T]):
        self.wrapped = wrapped

    def strategy_keys(self) -> List[str]:
        return self.wrapped.strategy_keys()

    def encode(self, encoder: IEncoder, documents: List[T]) -> Dict[str, npt.ArrayLike]:
        raw = self.wrapped.encode(encoder, documents)
        converted: dict[str, npt.ArrayLike] = {}
        for key, emb in raw.items():
            if isinstance(emb, torch.Tensor):
                converted[key] = emb.detach().cpu().numpy()
            else:
                converted[key] = emb
        return converted



# DatasetEncoder now composes encoder + dataset + two independent strategies
class DatasetEncoder(IDatasetEncoder, Generic[Q,T]):
    def __init__(
        self,
        encoder: IEncoder,
        dataset: IMultiLabelClassificationDataset[Q,T],
        query_strategy: Optional[IQueryEncoderStrategy[Q]] = None,
        document_strategy: Optional[IDocumentEncoderStrategy[T]] = None,
    ):
        self.encoder = encoder
        self.dataset = dataset
        # Provide sensible defaults when possible (text strategies)
        self.query_strategy = query_strategy
        self.document_strategy = document_strategy

    def encode_queries(self) -> Dict[str, npt.ArrayLike]:
        queries = self.dataset.get_queries()
        if self.query_strategy is None:
            raise RuntimeError("No query encoding strategy provided")
        return self.query_strategy.encode(self.encoder, queries)

    def encode_documents(self) -> Dict[str, npt.ArrayLike]:
        documents = self.dataset.get_documents()
        if self.document_strategy is None:
            raise RuntimeError("No document encoding strategy provided")
        return self.document_strategy.encode(self.encoder, documents)
    
    def get_dataset(self) -> IMultiLabelClassificationDataset[Q,T]:
        return self.dataset
    
    def get_encoder(self) -> IEncoder:
        return self.encoder


