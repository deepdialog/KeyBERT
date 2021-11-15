import re
import jieba
def gen_candidates_zh(docs, ngram_range=(1, 1)):
    sdocs = re.split(r'[。！；？，,.?：:、]', docs)
    res = set()
    for sdoc in sdocs:
        res
        cdoc = list(jieba.cut(re.sub('\W*', '', sdoc)))
        for i in range(ngram_range[0], ngram_range[1] + 1):
            for j in range(i, len(cdoc) + 1):
                res.add(''.join(cdoc[j-i:j]))
    return list(res)
