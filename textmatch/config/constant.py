
# 
import os
import threading

class Constant():

    PWD_PATH = './' #os.getenv("HOLMES_ROOT_PATH")
    LOG_PATH = os.path.join(PWD_PATH, 'logs/logs')


    base_dir = './data/'
    # 停用词
    STOPWORDS_FILE = os.path.join(base_dir,'./text_model_file/stop_words/stop_words.txt')

    # bow配置
    BOW_DIC_PATH = os.path.join(base_dir,'./text_model_file/bow_modelfile/ths_dict.dict')
    BOW_INDEX_PARH = os.path.join(base_dir,'./text_model_file/bow_modelfile/ths_bow.index')

    # tfidf配置
    TFIDF_DIC_PATH = os.path.join(base_dir,'./text_model_file/tfidf_modelfile/ths_dict.dict')
    TFIDF_MODEL_PATH = os.path.join(base_dir,'./text_model_file/tfidf_modelfile/ths_tfidf.model')
    TFIDF_INDEX_PATH = os.path.join(base_dir,'./text_model_file/tfidf_modelfile/ths_tfidf.index')

    # ngram_tfidf_modelfile
    NGRAM_TFIDF_DIC_PATH = os.path.join(base_dir,'./text_model_file/ngram_tfidf_modelfile/ths_dict.dict')
    NGRAM_TFIDF_MODEL_PATH = os.path.join(base_dir,'./text_model_file/ngram_tfidf_modelfile/ths_tfidf.model')
    NGRAM_TFIDF_INDEX_PATH = os.path.join(base_dir,'./text_model_file/ngram_tfidf_modelfile/ths_tfidf.index')

    # w2v配置
    W2V_MODEL_FILE = os.path.join(base_dir,'./text_model_file/word2vec_modelfile/word2vec_wx')

    # bert配置
    BERT_CONFIG_PATH = os.path.join(base_dir,'./text_model_file/simbert_modelfile/bert_config.json')
    BERT_CHECKPOINT_PATH = os.path.join(base_dir,'./text_model_file/simbert_modelfile/bert_model.ckpt')
    BERT_DICT_PATH = os.path.join(base_dir,'./text_model_file/simbert_modelfile/vocab.txt')
    
    # albert配置
    ALBERT_CONFIG_PATH = os.path.join(base_dir,'./text_model_file/albert_tiny_google_zh_489k/albert_config.json')
    # ALCHECKPOINT_PATH = os.path.join(base_dir,'./text_model_file/albert_tiny_google_zh_489k/albert_model.ckpt')
    ALBERT_DICT_PATH = os.path.join(base_dir,'./text_model_file/albert_tiny_google_zh_489k/vocab.txt')
    ALBERT_CHECKPOINT_PATH = os.path.join(base_dir, './text_model_file/albert_tiny_google_zh_489k/best_model.weights')


    # text资源位置
    TEXT_JSON_PATH = os.path.join(base_dir,'./text_model_file/text_words/text.json') 
    