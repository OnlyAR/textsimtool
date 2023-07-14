# @File Name:     textsimtool
# @Author :       Jun
# @date:          2023/7/13
# @Description :
from loguru import logger
from typing import List, Union, Dict
import numpy as np
import pandas as pd
from text2vec import SentenceModel

from textsimtool.utils.util import cos_sim, dot_score, semantic_search


class SimilarityABC:
    """
    Interface for similarity compute and search.

    In all instances, there is a corpus against which we want to perform the similarity search.
    For each similarity search, the input is a document or a corpus, and the output are the similarities
    to individual corpus documents.
    """

    def add_corpus(self, corpus: pd.DataFrame):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : pd.DataFrame
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute similarity between two texts.
        :param a: list of str or str
        :param b: list of str or str
        :param score_function: function to compute similarity, default cos_sim
        :return: similarity score, torch.Tensor, Matrix with res[i][j] = cos_sim(a[i], b[j])
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10):
        """
        Find the topn most similar texts to the query against the corpus.
        :param queries: Dict[str(query_id), str(query_text)] or List[str] or str
        :param topn: int
        :return: Dict[str, Dict[str, float]], {query_id: {corpus_id: similarity_score}, ...}
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")


class Similarity(SimilarityABC):
    """
    Sentence Similarity:
    1. Compute the similarity between two sentences
    2. Retrieves most similar sentence of a query against a corpus of documents.

    The index supports adding new documents dynamically.
    """

    def __init__(
            self,
            text_column: str = 'sentence',
            corpus: pd.DataFrame = None,
            model_name_or_path="shibing624/text2vec-base-chinese",
            encoder_type="MEAN",
            max_seq_length=128,
            device=None,
    ):
        """
        Initialize the similarity object.
        :param model_name_or_path: Transformer model name or path, like:
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'bert-base-uncased', 'bert-base-chinese',
             'shibing624/text2vec-base-chinese', ...
            model in HuggingFace Model Hub and release from https://github.com/shibing624/text2vec
        :param corpus: Corpus of documents to use for similarity queries.
        :param max_seq_length: Max sequence length for sentence model.
        """
        if isinstance(model_name_or_path, str):
            self.sentence_model = SentenceModel(
                model_name_or_path,
                encoder_type=encoder_type,
                max_seq_length=max_seq_length,
                device=device
            )
        else:
            raise ValueError("model_name_or_path is transformers model name or path")
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.corpus = pd.DataFrame()
        self.text_column = text_column
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: {self.sentence_model}"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: pd.DataFrame):
        """
        Extend the corpus with new documents.
        :param corpus: corpus of documents to use for similarity queries.
        :return: self.corpus, self.corpus embeddings
        """
        logger.info(f"Start computing corpus embeddings, new docs: {len(corpus)}")
        corpus_embeddings = self._get_vector(list(corpus[self.text_column]), show_progress_bar=True).tolist()
        corpus['embedding'] = corpus_embeddings
        print(corpus.head())
        self.corpus = pd.concat([self.corpus, corpus], ignore_index=True)
        self.corpus.reset_index(drop=True, inplace=True)
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}")

    def _get_vector(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 64,
            show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Returns the embeddings for a batch of sentences.
        :param sentences:
        :return:
        """
        return self.sentence_model.encode(sentences, batch_size=batch_size, show_progress_bar=show_progress_bar)

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]], score_function: str = "cos_sim"):
        """
        Compute similarity between two texts.
        :param a: list of str or str
        :param b: list of str or str
        :param score_function: function to compute similarity, default cos_sim
        :return: similarity score, torch.Tensor, Matrix with res[i][j] = cos_sim(a[i], b[j])
        """
        if score_function not in self.score_functions:
            raise ValueError(f"score function: {score_function} must be either (cos_sim) for cosine similarity"
                             " or (dot) for dot product")
        score_function = self.score_functions[score_function]
        text_emb1 = self._get_vector(a)
        text_emb2 = self._get_vector(b)

        return score_function(text_emb1, text_emb2)

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return 1 - self.similarity(a, b)

    def most_similar(self, query: str, topn: int = 10, score_function: str = "cos_sim", **kwargs):
        """
        Find the topn most similar texts to the queries against the corpus.
        :param query: str or list of str
        :param topn: int
        :param score_function: function to compute similarity, default cos_sim
        :return: Dict[str, Dict[str, float]], {query_id: {corpus_id: similarity_score}, ...}
        """
        if score_function not in self.score_functions:
            raise ValueError(f"score function: {score_function} must be either (cos_sim) for cosine similarity"
                             " or (dot) for dot product")
        score_function = self.score_functions[score_function]
        queries_embeddings = self._get_vector(query)
        corpus = self.corpus.copy()

        for col, val in kwargs.items():
            corpus = corpus[corpus[col] == val]

        corpus_embeddings = np.array(corpus['embedding'].tolist(), dtype=np.float32)
        hits = semantic_search(queries_embeddings, corpus_embeddings, top_k=topn, score_function=score_function)[0]

        for hit in hits:
            corpus.loc[hit['corpus_id'], 'score'] = hit['score']

        corpus.sort_values(by=['score'], ascending=False, inplace=True)
        result = corpus.head(topn)
        result.reset_index(drop=True, inplace=True)
        result.drop(columns=['embedding'], inplace=True)
        return result

    def save_index(self, index_path: str = "corpus_emb.csv"):
        """
        Save corpus embeddings to json file.
        :param index_path: json file path
        :return:
        """
        self.corpus.to_csv(f'{index_path}', index=False)
        logger.debug(f"Save corpus embeddings to file: {index_path}.")

    def load_index(self, index_path: str = "corpus_emb.json"):
        """
        Load corpus embeddings from json file.
        :param index_path: json file path
        :return: list of corpus embeddings, dict of corpus ids map, dict of corpus
        """
        try:
            corpus = pd.read_csv(index_path)
            self.corpus = pd.concat([self.corpus, corpus], ignore_index=True)
            self.corpus.reset_index(drop=True, inplace=True)
            logger.debug(f"Load corpus embeddings from file: {index_path}.")
        except FileNotFoundError:
            logger.error("Error: Could not load corpus embeddings from file.")
