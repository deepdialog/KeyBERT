[![PyPI - Python](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue.svg)](https://pypi.org/project/keybert/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MaartenGr/keybert/blob/master/LICENSE)
[![Build](https://img.shields.io/github/workflow/status/MaartenGr/keyBERT/Code%20Checks/master)](https://pypi.org/project/keybert/)

# ZhKeyBERT

Based on [KeyBERT](https://github.com/MaartenGr/KeyBERT), enhance the keyword
extraction model for the characteristics of Chinese.

Corresponding medium post can be found [here](https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea).

<a name="toc"/></a>
## Table of Contents  
<!--ts-->
   1. [About the Project](#about)  
   2. [Getting Started](#gettingstarted)    
        2.1. [Installation](#installation)    
        2.2. [Basic Usage](#usage)     
        2.3. [Max Sum Similarity](#maxsum)  
        2.4. [Maximal Marginal Relevance](#maximal)  
        2.5. [Embedding Models](#embeddings)
<!--te-->


<a name="about"/></a>
## 1. About the Project
[Back to ToC](#toc)  

Although there are already many methods available for keyword generation 
(e.g., 
[Rake](https://github.com/aneesha/RAKE), 
[YAKE!](https://github.com/LIAAD/yake), TF-IDF, etc.) 
I wanted to create a very basic, but powerful method for extracting keywords and keyphrases. 
This is where **KeyBERT** comes in! Which uses BERT-embeddings and simple cosine similarity
to find the sub-phrases in a document that are the most similar to the document itself.

First, document embeddings are extracted with BERT to get a document-level representation. 
Then, word embeddings are extracted for N-gram words/phrases. Finally, we use cosine similarity 
to find the words/phrases that are the most similar to the document. The most similar words could 
then be identified as the words that best describe the entire document.  

KeyBERT is by no means unique and is created as a quick and easy method
for creating keywords and keyphrases. Although there are many great 
papers and solutions out there that use BERT-embeddings 
(e.g., 
[1](https://github.com/pranav-ust/BERT-keyphrase-extraction),
[2](https://github.com/ibatra/BERT-Keyword-Extractor),
[3](https://www.preprints.org/manuscript/201908.0073/download/final_file),
), I could not find a BERT-based solution that did not have to be trained from scratch and
could be used for beginners (**correct me if I'm wrong!**).
Thus, the goal was a `pip install keybert` and at most 3 lines of code in usage.   

<a name="gettingstarted"/></a>
## 2. Getting Started
[Back to ToC](#toc)  

<a name="installation"/></a>
###  2.1. Installation

<a name="usage"/></a>
###  2.2. Usage

The most minimal example can be seen below for the extraction of keywords:
```python
from keybert import KeyBERT, extract_kws

docs = """时值10月25日抗美援朝纪念日，《长津湖》片方发布了“纪念中国人民志愿军抗美援朝出国作战71周年特别短片”，再次向伟大的志愿军致敬！
电影《长津湖》全情全景地还原了71年前抗美援朝战场上那场史诗战役，志愿军奋不顾身的英勇精神令观众感叹：“岁月峥嵘英雄不灭，丹心铁骨军魂永存！”影片上映以来票房屡创新高，目前突破53亿元，暂列中国影史票房总榜第三名。
值得一提的是，这部影片的很多主创或有军人的血脉，或有当兵的经历，或者家人是军人。提起这些他们也充满自豪，影片总监制黄建新称：“当兵以后会有一种特别能坚持的劲儿。”饰演雷公的胡军透露：“我父亲曾经参加过抗美援朝，还得了一个三等功。”影片历史顾问王树增表示：“我当了五十多年的兵，我的老部队就是上甘岭上下来的，那些老兵都是我的偶像。”
“身先士卒卫华夏家国，血战无畏护山河无恙。”片中饰演七连连长伍千里的吴京感叹：“要永远记住这些先烈们，他们给我们带来今天的和平。感谢他们的付出，才让我们有今天的幸福生活。”饰演新兵伍万里的易烊千玺表示：“战争的残酷、碾压式的伤害，其实我们现在的年轻人几乎很难能体会到，希望大家看完电影后能明白，是那些先辈们的牺牲奉献，换来了我们的现在。”
影片对战争群像的恢弘呈现，对个体命运的深切关怀，令许多观众无法控制自己的眼泪，观众称：“当看到影片中的惊险战斗场面，看到英雄们壮怀激烈的拼杀，为国捐躯的英勇无畏和无悔付出，我明白了为什么说今天的幸福生活来之不易。”（记者 王金跃）"""
kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
extract_kws_zh(docs, kw_model)
```

Comparison
```python
>>> extract_kws_zh(docs, kw_model)

[('纪念中国人民志愿军抗美援朝', 0.7034),
 ('电影长津湖', 0.6285),
 ('周年特别短片', 0.5668),
 ('纪念中国人民志愿军', 0.6894),
 ('作战71周年', 0.5637)]
>>> import jieba; kw_model.extract_keywords(' '.join(jieba.cut(docs)), keyphrase_ngram_range=(1, 3), 
                                            use_mmr=True, diversity=0.25)

[('抗美援朝 纪念日 长津湖', 0.796),
 ('纪念 中国人民志愿军 抗美援朝', 0.7577),
 ('作战 71 周年', 0.6126),
 ('25 抗美援朝 纪念日', 0.635),
 ('致敬 电影 长津湖', 0.6514)]
```

You can set `ngram_range`, whose default value is `(1, 3)`,
to set the length of the resulting keywords/keyphrases:

```python
>>> extract_kws_zh(docs, kw_model, ngram_range=(1, 1))
[('中国人民志愿军', 0.6094),
 ('长津湖', 0.505),
 ('周年', 0.4504),
 ('影片', 0.447),
 ('英雄', 0.4297)]
```

```python
>>> extract_kws_zh(docs, kw_model, ngram_range=(1, 2))
[('纪念中国人民志愿军', 0.6894),
 ('电影长津湖', 0.6285),
 ('年前抗美援朝', 0.476),
 ('中国人民志愿军抗美援朝', 0.6349),
 ('中国影史', 0.5534)]
``` 

**NOTE**: For a full overview of all possible transformer models see [sentence-transformer](https://www.sbert.net/docs/pretrained_models.html).
I would advise `"paraphrase-multilingual-MiniLM-L12-v2"` Chinese documents for efficiency
and acceptable accuracy.

<a name="maximal"/></a>
###  2.3. Maximal Marginal Relevance

It's recommended to use Maximal Margin Relevance (MMR) for diversity by
setting the optional parameter `use_mmr`, which is `True` in default.  
To diversify the results, we can use MMR to create
keywords / keyphrases which is also based on cosine similarity. The results 
with **high diversity**:

```python
>>> extract_kws_zh(docs, kw_model, use_mmr = True, diversity=0.7)

[('纪念中国人民志愿军抗美援朝', 0.7034),
 ('观众无法控制自己', 0.1212),
 ('山河无恙', 0.2233),
 ('影片上映以来', 0.5427),
 ('53亿元', 0.3287)]
``` 

The results with **low diversity**:  

```python
>>> kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english', 
                              use_mmr=True, diversity=0.2)
[('纪念中国人民志愿军抗美援朝', 0.7034),
 ('电影长津湖', 0.6285),
 ('纪念中国人民志愿军', 0.6894),
 ('周年特别短片', 0.5668),
 ('作战71周年', 0.5637)]
``` 

And the default and recommended `diversity` is `0.25`.

<a name="embeddings"/></a>
###  2.4. Embedding Models
KeyBERT supports many embedding models that can be used to embed the documents and words:

* Sentence-Transformers
* Flair
* Spacy
* Gensim
* USE

Click [here](https://maartengr.github.io/KeyBERT/guides/embeddings.html) for a full overview of all supported embedding models.

**Sentence-Transformers**  
You can select any model from `sentence-transformers` [here](https://www.sbert.net/docs/pretrained_models.html) 
and pass it through KeyBERT with `model`:

```python
from keybert import KeyBERT
kw_model = KeyBERT(model='all-MiniLM-L6-v2')
```

Or select a SentenceTransformer model with your own parameters:

```python
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=sentence_model)
```

**Flair**  
[Flair](https://github.com/flairNLP/flair) allows you to choose almost any embedding model that 
is publicly available. Flair can be used as follows:

```python
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings

roberta = TransformerDocumentEmbeddings('roberta-base')
kw_model = KeyBERT(model=roberta)
```

You can select any 🤗 transformers model [here](https://huggingface.co/models).


## Citation
To cite KeyBERT in your work, please use the following bibtex reference:

```bibtex
@misc{grootendorst2020keybert,
  author       = {Maarten Grootendorst},
  title        = {KeyBERT: Minimal keyword extraction with BERT.},
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.3.0},
  doi          = {10.5281/zenodo.4461265},
  url          = {https://doi.org/10.5281/zenodo.4461265}
}
```

## References
Below, you can find several resources that were used for the creation of KeyBERT 
but most importantly, these are amazing resources for creating impressive keyword extraction models: 

**Papers**:  
* Sharma, P., & Li, Y. (2019). [Self-Supervised Contextual Keyword and Keyphrase Retrieval with Self-Labelling.](https://www.preprints.org/manuscript/201908.0073/download/final_file)

**Github Repos**:  
* https://github.com/thunlp/BERT-KPE
* https://github.com/ibatra/BERT-Keyword-Extractor
* https://github.com/pranav-ust/BERT-keyphrase-extraction
* https://github.com/swisscom/ai-research-keyphrase-extraction

**MMR**:  
The selection of keywords/keyphrases was modeled after:
* https://github.com/swisscom/ai-research-keyphrase-extraction

**NOTE**: If you find a paper or github repo that has an easy-to-use implementation
of BERT-embeddings for keyword/keyphrase extraction, let me know! I'll make sure to
add a reference to this repo. 

