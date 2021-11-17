[![PyPI - Python](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue.svg)](https://pypi.org/project/keybert/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MaartenGr/keybert/blob/master/LICENSE)

# ZhKeyBERT

基于[KeyBERT](https://github.com/MaartenGr/KeyBERT), 增强模型对中文
关键词提取的支持。

## 目录  
<!--ts-->
   1. [开始](#gettingstarted)    
        1.1. [安装](#installation)     
        1.2. [用法](#usage)    
        1.3. [对比](#compare)    
        1.4. [模型](#embeddings)    
<!--te-->

<a name="installation"/></a>
###  1.1. 安装

```
pip install zhkeybert --user
```

<a name="usage"/></a>
###  1.2. 用法

最简单的样例：

```python
from zhkeybert import KeyBERT, extract_kws_zh

docs = """时值10月25日抗美援朝纪念日，《长津湖》片方发布了“纪念中国人民志愿军抗美援朝出国作战71周年特别短片”，再次向伟大的志愿军致敬！
电影《长津湖》全情全景地还原了71年前抗美援朝战场上那场史诗战役，志愿军奋不顾身的英勇精神令观众感叹：“岁月峥嵘英雄不灭，丹心铁骨军魂永存！”影片上映以来票房屡创新高，目前突破53亿元，暂列中国影史票房总榜第三名。
值得一提的是，这部影片的很多主创或有军人的血脉，或有当兵的经历，或者家人是军人。提起这些他们也充满自豪，影片总监制黄建新称：“当兵以后会有一种特别能坚持的劲儿。”饰演雷公的胡军透露：“我父亲曾经参加过抗美援朝，还得了一个三等功。”影片历史顾问王树增表示：“我当了五十多年的兵，我的老部队就是上甘岭上下来的，那些老兵都是我的偶像。”
“身先士卒卫华夏家国，血战无畏护山河无恙。”片中饰演七连连长伍千里的吴京感叹：“要永远记住这些先烈们，他们给我们带来今天的和平。感谢他们的付出，才让我们有今天的幸福生活。”饰演新兵伍万里的易烊千玺表示：“战争的残酷、碾压式的伤害，其实我们现在的年轻人几乎很难能体会到，希望大家看完电影后能明白，是那些先辈们的牺牲奉献，换来了我们的现在。”
影片对战争群像的恢弘呈现，对个体命运的深切关怀，令许多观众无法控制自己的眼泪，观众称：“当看到影片中的惊险战斗场面，看到英雄们壮怀激烈的拼杀，为国捐躯的英勇无畏和无悔付出，我明白了为什么说今天的幸福生活来之不易。”（记者 王金跃）"""
kw_model = KeyBERT()
extract_kws_zh(docs, kw_model)
```

`ngram_range`决定了结果短句可以由多少个词语构成

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

`use_mmr`默认为`True`，会使用Maximal Marginal Relevance（MMR）算法
增加结果的多样性，`diversity`（[0,1]之间,默认`0.25`）控制了多样性的程度，
值越大程度越高。若`use_mmr=False`，则容易出现多个结果包含同一个词语的情况。

高多样性的结果很杂乱：
```python
>>> extract_kws_zh(docs, kw_model, use_mmr=True, diversity=0.7)
[('纪念中国人民志愿军抗美援朝', 0.7034),
 ('观众无法控制自己', 0.1212),
 ('山河无恙', 0.2233),
 ('影片上映以来', 0.5427),
 ('53亿元', 0.3287)]
``` 

低多样性的结果重复度相对较高：
```python
>>> extract_kws_zh(docs, kw_model, use_mmr=True, diversity=0.7)
[('纪念中国人民志愿军抗美援朝', 0.7034),
 ('电影长津湖', 0.6285),
 ('纪念中国人民志愿军', 0.6894),
 ('周年特别短片', 0.5668),
 ('作战71周年', 0.5637)]
``` 

`diversity=0.0`的结果与`use_mmr=False`的结果无异，权衡两种极端，比较推荐默认值`diversity=0.25`.
<a name="compare"/></a>
### 对比
本项目对KeyBERT的主要改进有：
- 细化候选关键词的筛选，避免跨句组合等情况
- 调整超参数，寻找效果较优的组合（例如原始模型中`use_maxsum`的效果奇差）
- 找出效率和效果均比较优秀的模型`paraphrase-multilingual-MiniLM-L12-v2`

```python
>>> from zhkeybert import KeyBERT, extract_kws_zh
>>> import jieba
>>> lines = []
>>> with open('tests/example1.txt', 'r') as f:
        for line in f:
            lines.append(line)
>>> kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
>>> for line in lines: 
        print(extract_kws_zh(line, kw_model))
[('网络文明大会', 0.7627), ('推进文明办网', 0.7084), ('北京国家会议', 0.5802), ('文明办网', 0.7105), ('大会主题为', 0.6182)]
[('国家自然科学奖评选出', 0.7038), ('前沿研究领域', 0.6102), ('的重要科学', 0.62), ('自然科学奖', 0.693), ('自然科学奖一等奖', 0.6887)]
[('等蔬菜价格下降', 0.7361), ('滑县瓜菜种植', 0.649), ('蔬菜均价每公斤', 0.6768), ('全国蔬菜均价', 0.709), ('村蔬菜种植', 0.6536)]
[('中国共产党的任务', 0.7928), ('一个中国原则', 0.751), ('中国共产党的', 0.7541), ('统一是中国', 0.7095), ('中国人民捍卫', 0.7081)]
>>> for line in lines:
        print(kw_model.extract_keywords(' '.join(jieba.cut(line)), keyphrase_ngram_range=(1, 3), 
                                        use_mmr=True, diversity=0.25))
[('中国 网络 文明', 0.7355), ('网络 文明 大会', 0.7018), ('北京 国家 会议', 0.6802), ('首届 中国 网络', 0.723), ('打造 我国 网络', 0.6766)]
[('基础 研究 国家自然科学奖', 0.7712), ('领域 重要 科学', 0.7054), ('研究 国家自然科学奖 评选', 0.7441), ('研究 国家自然科学奖', 0.7499), ('自然科学 一等奖', 0.7193)]
[('目前 蔬菜 储备', 0.8036), ('滑县 瓜菜 种植', 0.7484), ('大省 蔬菜 面积', 0.798), ('居民 蔬菜 供应', 0.7902), ('设施 蔬菜 大省', 0.792)]
[('统一 中国 全体', 0.7614), ('谈到 祖国统一', 0.7532), ('习近平 总书记 表示', 0.6338), ('祖国统一 问题 总书记', 0.7368), ('中国共产党 任务 实现', 0.679)]
```

<a name="embeddings"/></a>
###  1.4. 模型
KeyBERT支持许多embedding模型，但是对于中文语料，应该采用带有`multilingual`的模型，
默认模型为`paraphrase-multilingual-MiniLM-L12-v2`，也可手动指定：`KeyBERT(model=...)`。

可以从KeyBERT的[文档](https://maartengr.github.io/KeyBERT/guides/embeddings.html)中
了解如何加载各种来源的模型

下面是一些测试的结果

|模型名称|计算速度|精确度|大小|
|------|-------|-----|------|
|universal-sentence-encoder-multilingual-large|很慢|较好|近1G|
|universal-sentence-encoder-multilingual|一般|一般|小几百M|
|paraphrase-multilingual-MiniLM-L12-v2|快|较好|400M|
|bert-base-multilingual-cased (Flair)|慢|一般|几M|
