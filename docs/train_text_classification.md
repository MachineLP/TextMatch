> 文本分类模型的输入是 textembedding模块提取的vector。

### run train_model/ (train embedding(bow/tfidf/ngram tfidf/bert/albert...  train classifer))
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python train_model/train_bow.py                 (文本embedding)
python train_model/train_tfidf.py               (文本embedding)
python train_model/train_ngram_tfidf.py         (文本embedding)
python train_model/train_bert.py                (文本embedding)
python train_model/train_albert.py              (文本embedding)
python train_model/train_w2v.py                 (文本embedding)
python train_model/train_dssm.py                (文本embedding)
python train_model/train_lr_classifer.py             (文本分类)
python train_model/train_gbdt_classifer.py           (文本分类)
python train_model/train_gbdlr_classifer.py          (文本分类)
python train_model/train_lgb_classifer.py            (文本分类)
python train_model/train_xgb_classifer.py            (文本分类)
python train_model/train_dnn_classifer.py            (文本分类)
```


