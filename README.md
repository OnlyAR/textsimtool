# Textsimtool 文本相似度计算工具

## 1. 介绍

短文本相似度计算是 NLP 领域的一个经典问题，现有的相似度计算工具[shibing624/similarities](https://github.com/shibing624/similarities)已经能达到不错的效果，但是难以满足本人的要求。因此我开发了这个工具，添加了按标签筛选的功能。

## 2. 使用方法

```python
import pandas as pd
import textsimtool

sim = textsimtool.Similarity(model_name_or_path='bert-base-chinese', text_column='input')

data = pd.DataFrame({
    'input': ['我爱你', '我喜欢你', '我恨你', '我讨厌你'],
    'label': ['p', 'p', 'n', 'n']
})

sim.add_corpus(data)  # add corpus

"""calculate similarity"""
print(sim.distance('我爱你', '我喜欢你'))  # return float

"""search similar sentence"""
print(sim.most_similar('我爱死你了', topn=2))   # return DataFrame

print(sim.most_similar('我爱死你了', topn=2, label="p"))  # filter by column


"""save and load index"""
sim.save_index('example.index')  # save index
sim.load_index('example.index')  # load index
```