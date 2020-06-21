
TextMatch

TextMatch is a semantic matching model library for QA & text search ...  It's easy to train models and to export representation vectors.

### run examples
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
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



### run tests/core_test
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python tests/core_test/qa_match_test.py
python tests/core_test/text_embedding_test.py
```




### run tests/models_test
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python tests/models_test/bm25_test.py
python tests/models_test/edit_sim_test.py
python tests/models_test/jaccard_sim_test.py
python tests/models_test/bow_sklearn_test.py
python tests/models_test/tf_idf_sklearn_test.py
python tests/models_test/ngram_tf_idf_sklearn_test.py
python tests/models_test/w2v_test.py
```


