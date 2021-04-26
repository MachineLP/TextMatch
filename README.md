
TextMatch

TextMatch is a semantic matching model library for QA & text search ...  It's easy to train models and to export representation vectors.

Let's [**Run examples**](./EXAMPLES.md) !

## test models List

|         Model       |   models   |    tests   |  
| :-----------------: | :---------:| :---------:|
|  Bow  | [1](./textmatch/models/text_embedding/bow_sklearn.py)                    | [test](./tests/models_test/bow_sklearn_test.py) |
| TFIDF | [2](./textmatch/models/text_embedding/tf_idf_sklearn.py)                 | [test](./tests/models_test/tf_idf_sklearn_test.py) |
| Ngram-TFIDF     | [3](./textmatch/models/text_embedding/ngram_tf_idf_sklearn.py) | [test](./tests/models_test/ngram_tf_idf_sklearn_test.py) |
| W2V     | [4](./textmatch/models/text_embedding/w2v.py)                          | [test](./tests/models_test/w2v_test.py) |
| BERT    | [5](./textmatch/models/text_embedding/bert_embedding.py)               | [BERT-whitening](https://github.com/bojone/BERT-whitening)校正了BERT句向量分布，使cos相似度更合理。 [SimCSE](https://github.com/bojone/SimCSE)|
| ALBERT  | [6](./textmatch/models/text_embedding/albert_embedding.py)             | [test](./tests/models_test/albert_embedding_test.py) 链接:https://pan.baidu.com/s/1HSVS104iBBOsfw7hXdyqLQ  密码:808k |
| DSSM    |  |  |
| bm25    | [8](./textmatch/models/text_search/bm25.py)  | [test](./tests/models_test/bm25_test.py) |
| edit_sim    | [9](./textmatch/models/text_search/edit_sim.py)  | [test](./tests/models_test/edit_sim_test.py) |
| jaccard_sim    | [10](./textmatch/models/text_search/jaccard_sim.py)  | [test](./tests/models_test/jaccard_sim_test.py) |
| wmd    | [11](./tests/models_test/wmd_test.py)  | [test](./tests/models_test/wmd_test.py) |
| Kmeans    | [12](./textmatch/tools/clustering/kmeans.py)  | [test](./tests/tools_test/kmeans_test.py) |
| DBSCAN    | [13](./textmatch/tools/clustering/dbscan.py)  | [test](./tests/tools_test/dbscan_test.py) |
| PCA    | [14](./textmatch/tools/decomposition/pca.py)  | [test](./tests/tools_test/pca_test.py) |
| FAISS  | [15](./textmatch/tools/faiss/faiss.py)  | [test](./tests/tools_test/faiss_test.py) |
| ....    |  |  |
| lr    | [92](./textmatch/models/ml/lr.py) | [test](./tests/ml_test/lr_test.py) |
| gbdt    | [93](./textmatch/models/ml/gbdt.py) | [test](./tests/ml_test/gbdt_test.py) |
| gbdt_lr    | [94](./textmatch/models/ml/gbdt_lr.py) | [test](./tests/ml_test/gbdt_lr_test.py) |
| lgb    | [95](./textmatch/models/ml/lgb.py) | [test](./tests/ml_test/lgb_test.py) |
| xgb    | [96](./textmatch/models/ml/xgb.py) | [test](./tests/ml_test/xgb_test.py) |
| Bagging    | [97](./textmatch/models/text_embedding/model_factory_sklearn.py)     | [test](./tests/models_test/factory_test.py)  |
| QA    | [98](./textmatch/core/qa_match.py)     | [test](./tests/core_test/qa_match_test.py)  |
| Text Embedding    | [99](./textmatch/core/text_embedding.py)     | [test](./tests/core_test/text_embedding_test.py)  |

# 

## train models List
|         Model       |   models   |    train   |  
| :-----------------: | :---------:| :---------:|
|  Bow  | [1](./textmatch/models/text_embedding/bow_sklearn.py)                    | [train](./train_model/train_bow.py) |
| TFIDF | [2](./textmatch/models/text_embedding/tf_idf_sklearn.py)                 | [train](./train_model/train_tfidf.py) |
| Ngram-TFIDF     | [3](./textmatch/models/text_embedding/ngram_tf_idf_sklearn.py) | [train](./train_model/train_ngram_tfidf.py) |
| W2V     | [4](./textmatch/models/text_embedding/w2v.py)                          | [train](./train_model/train_w2v.py) |
| BERT    | [5](./textmatch/models/text_embedding/bert_embedding.py)               | [train](./train_model/train_bert.py) |
| ALBERT  | [6](./textmatch/models/text_embedding/albert_embedding.py)             | [train](./train_model/train_albert.py) |
| DSSM    |  |  |
| Kmeans    | [12](./textmatch/tools/clustering/kmeans.py)  | [train](./tests/tools_test/kmeans_test.py) |
| DBSCAN    | [13](./textmatch/tools/clustering/dbscan.py)  | [train](./tests/tools_test/dbscan_test.py) |
| PCA    | [14](./textmatch/tools/decomposition/pca.py)  | [train](./tests/tools_test/pca_test.py) |
| ....    |  |  |
| lr    | [92](./textmatch/models/ml/lr.py) | [train](./train_model/train_lr_classifer.py) |
| gbdt    | [93](./textmatch/models/ml/gbdt.py) | [train](./train_model/train_gbdt_classifer.py) |
| gbdt_lr    | [94](./textmatch/models/ml/gbdt_lr.py) | [train](./train_model/train_gbdtlr_classifer.py) |
| lgb    | [95](./textmatch/models/ml/lgb.py) | [train](./train_model/train_lgb_classifer.py) |
| xgb    | [96](./textmatch/models/ml/xgb.py) | [train](./train_model/train_xgb_classifer.py) |




## TODO
（1）knn
（2）dssm
（3）[实体识别](https://github.com/bojone/bert4keras/blob/master/examples/task_sequence_labeling_ner_crf.py)
（4）[文本纠错](https://github.com/shibing624/pycorrector)



# 
- wechat ID: lp9628




# 
# 


### 样例：
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python examples/text_search.py

```

examples/text_search.py
```python
import sys
from textmatch.models.text_embedding.model_factory_sklearn import ModelFactory


if __name__ == '__main__':
    # doc
    doc_dict = {"0":"我去玉龙雪山并且喜欢玉龙雪山玉龙雪山", "1":"我在玉龙雪山并且喜欢玉龙雪山", "2":"我在九寨沟", "3":"你好"}   
    # query
    query = "我在九寨沟,很喜欢"
    
    # 模型工厂，选择需要的模型加到列表中: 'bow', 'tfidf', 'ngram_tfidf', 'bert', 'albert', 'w2v'
    mf = ModelFactory( match_models=['bow', 'tfidf', 'ngram_tfidf'] )
    # 模型处理初始化
    mf.init(words_dict=doc_dict, update=True)

    # query 与 doc的相似度
    search_res = mf.predict(query)
    print ('search_res>>>>>', search_res) 
    # search_res>>>>> {'bow': [('0', 0.2773500981126146), ('1', 0.5303300858899106), ('2', 0.8660254037844388), ('3', 0.0)], 'tfidf': [('0', 0.2201159065358879), ('1', 0.46476266418455736), ('2', 0.8749225357988296), ('3', 0.0)], 'ngram_tfidf': [('0', 0.035719486884261346), ('1', 0.09654705406841395), ('2', 0.9561288696241232), ('3', 0.0)]}
    
    # query的embedding
    query_emb = mf.predict_emb(query)
    print ('query_emb>>>>>', query_emb) 
    '''
    pre_emb>>>>> {'bow': array([1., 0., 0., 1., 1., 0., 1., 0.]), 'tfidf': array([0.61422608, 0.        , 0.        , 0.4842629 , 0.4842629 ,
       0.        , 0.39205255, 0.        ]), 'ngram_tfidf': array([0.        , 0.        , 0.37156534, 0.37156534, 0.        ,
       0.        , 0.        , 0.29294639, 0.        , 0.37156534,
       0.37156534, 0.        , 0.        , 0.37156534, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.29294639, 0.37156534, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        ])}
    '''

```

# 

<img src="https://github.com/MachineLP/TextMatch/blob/master/docs/text_match.png" width="400" height="400" alt="text_match"/>

### 样例：
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python tests/tools_test/faiss_test.py

```

tests/tools_test/faiss_test.py
```python
import sys
import json 
import time
import faiss
import numpy as np
from faiss import normalize_L2
from textmatch.config.constant import Constant as const
from textmatch.core.text_embedding import TextEmbedding
from textmatch.tools.decomposition.pca import PCADecomposition
from textmatch.tools.faiss.faiss import FaissSearch

test_dict = {"id0": "其实事物发展有自己的潮流和规律",
   "id1": "当你身处潮流之中的时候，要紧紧抓住潮流的机会",
   "id2": "想办法脱颖而出，即使没有成功，也会更加洞悉时代的脉搏",
   "id3": "收获珍贵的知识和经验。而如果潮流已经退去",
   "id4": "这个时候再去往这个方向上努力，只会收获迷茫与压抑",
   "id5": "对时代、对自己都没有什么帮助",
   "id6": "但是时代的浪潮犹如海滩上的浪花，总是一浪接着一浪，只要你站在海边，身处这个行业之中，下一个浪潮很快又会到来。你需要敏感而又深刻地去观察，略去那些浮躁的泡沫，抓住真正潮流的机会，奋力一搏，不管成败，都不会遗憾。",
   "id7": "其实事物发展有自己的潮流和规律",
   "id8": "当你身处潮流之中的时候，要紧紧抓住潮流的机会" }


if __name__ == '__main__':
    # ['bow', 'tfidf', 'ngram_tfidf', 'bert']
    # ['bow', 'tfidf', 'ngram_tfidf', 'bert', 'w2v']
    # text_embedding = TextEmbedding( match_models=['bow', 'tfidf', 'ngram_tfidf', 'w2v'], words_dict=test_dict ) 
    text_embedding = TextEmbedding( match_models=['bow', 'tfidf', 'ngram_tfidf', 'w2v'], words_dict=None, update=False ) 
    feature_list = []
    for sentence in test_dict.values():
        pre = text_embedding.predict(sentence)
        feature = np.concatenate([pre[model] for model in ['bow', 'tfidf', 'ngram_tfidf', 'w2v']], axis=0)
        feature_list.append(feature)
    pca = PCADecomposition(n_components=8)
    data = np.array( feature_list )
    pca.fit( data )
    res = pca.transform( data )
    print('res>>', res)

   

    pre = text_embedding.predict("潮流和规律")
    feature = np.concatenate([pre[model] for model in ['bow', 'tfidf', 'ngram_tfidf', 'w2v']], axis=0)
    test = pca.transform( [feature] )

    faiss_search = FaissSearch( res, sport_mode=False )
    faiss_res = faiss_search.predict( test )
    print( "faiss_res:", faiss_res )
    '''
    faiss kmeans result times 8.0108642578125e-05
    faiss_res: [{0: 0.7833399, 7: 0.7833399, 3: 0.63782495}]
    '''

    
    faiss_search = FaissSearch( res, sport_mode=True )
    faiss_res = faiss_search.predict( test )
    print( "faiss_res:", faiss_res )
    '''
    faiss kmeans result times 3.266334533691406e-05
    faiss_res: [{0: 0.7833399, 7: 0.7833399, 3: 0.63782495}]
    '''
```

# 

### run train_model/ (train embedding(bow/tfidf/ngram tfidf/bert/albert...  train classifer))
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python train_model/train_bow.py                          (文本embedding)
python train_model/train_tfidf.py                        (文本embedding)
python train_model/train_ngram_tfidf.py                  (文本embedding)
python train_model/train_bert.py                         (文本embedding)
python train_model/train_albert.py                       (文本embedding)
python train_model/train_w2v.py                          (文本embedding)
python train_model/train_dssm.py                         (文本embedding)
python train_model/train_lr_classifer.py                 (文本分类)
python train_model/train_gbdt_classifer.py               (文本分类)
python train_model/train_gbdlr_classifer.py              (文本分类)
python train_model/train_lgb_classifer.py                (文本分类)
python train_model/train_xgb_classifer.py                (文本分类)
python train_model/train_dnn_classifer.py                (文本分类)
```

### run tests/core_test （qa/文本embedding）
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python tests/core_test/qa_match_test.py
python tests/core_test/text_embedding_test.py
```



### run tests/models_test （模型测试）
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python tests/models_test/bm25_test.py                    (bm25)
python tests/models_test/edit_sim_test.py                (编辑距离)
python tests/models_test/jaccard_sim_test.py             (jaccard)
python tests/models_test/bow_sklearn_test.py             (bow)
python tests/models_test/tf_idf_sklearn_test.py          (tf_idf)
python tests/models_test/ngram_tf_idf_sklearn_test.py    (ngram_tf_idf)
python tests/models_test/w2v_test.py                     (w2v)
python tests/models_test/albert_test.py                  (bert)
```

### run tests/ml_test  （机器学习模型测试）
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python tests/ml_test/lr_test.py                          (lr)
python tests/ml_test/gbdt_test.py                        (gbdt)
python tests/ml_test/gbdt_lr_test.py                     (gbdt_lr)
python tests/ml_test/lgb_test.py                         (lgb)
python tests/ml_test/xgb_test.py                         (xgb)
```

### run tests/tools_test   （聚类/降维工具测试）
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python tests/tools_test/kmeans_test.py                   (kmeans)
python tests/tools_test/dbscan_test.py                   (dbscan)
python tests/tools_test/pca_test.py                      (pca)
python tests/tools_test/faiss_test.py                    (faiss)
```

### run tests/tools_test   （词云）
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
cd tests/tools_test
python generate_word_cloud.py
```

<img src="https://github.com/MachineLP/TextMatch/blob/master/docs/pics/word_cloud.png" width="300" height="200" alt="word_cloud"/>

