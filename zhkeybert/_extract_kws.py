from typing import List, Tuple, Union
import re
import jieba

from ._model import KeyBERT


def gen_candidates_zh(docs: str, ngram_range: Tuple[int, int]) -> List[str]:
    """split the Chinese document into keyword candidates

    Args:
        docs (str): the Chinese document
        ngram_range (Tuple[int, int]): Length, in words, of the extracted keywords/keyphrases

    Returns:
        List[str]: keyword candidates
    """
    sdocs = re.split(r'[。！；？，,.?：:、“”]', docs)
    res = set()
    for sdoc in sdocs:
        res
        cdoc = list(jieba.cut(re.sub('\W*', '', sdoc)))
        for i in range(ngram_range[0], ngram_range[1] + 1):
            for j in range(i, len(cdoc) + 1):
                res.add(''.join(cdoc[j-i:j]))
    return list(res)


def extract_kws_zh(docs: str, model: KeyBERT,
                   ngram_range: Tuple[int, int] = (1, 3),
                   top_n: int = 5,
                   use_mmr: bool = True,
                   diversity: float = 0.25,) -> Union[List[Tuple[str, float]],
                                                      List[List[Tuple[str, float]]]]:
    """extract keywords from Chinese document

    Args:
        docs (str): the Chinese document
        model (keybert.KeyBERT): the KeyBERT model to do extraction
        ngram_range (Tuple[int, int], optional): Length, in words, of the extracted 
                        keywords/keyphrases. Defaults to (1, 3).
        top_n (int, optional): extract n keywords. Defaults to 5.
        use_mmr (bool, optional): Whether to use MMR. Defaults to True.
        diversity (float, optional): The diversity of results between 0 and 1 
                        if use_mmr is True. Defaults to 0.25.
    Returns:
        Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]: the top n keywords for a document
    """

    candi = gen_candidates_zh(docs, ngram_range)
    return model.extract_keywords(docs, candi,
                                  stop_words=None,
                                  top_n=top_n,
                                  use_mmr=use_mmr,
                                  diversity=diversity)
