
# 
import os
import yaml
import threading
import numpy as np
from easydict import EasyDict as edict

# 创建dict
__C = edict()
cfg = __C

# 定义配置dict
__C.emb = edict()
__C.emb.JIEBA_FLAG = True
__C.emb.DEL_STOPWORD = False
__C.emb.MAX_DF = 0.8
__C.emb.MIN_DF = 0.2
__C.emb.MAX_FEATURES = None
__C.emb.NGRAM_RANGE = 3


# ML
# LR / GBDTLR
__C.lr = edict()
__C.lr.max_iter=100

# GBDT / GBDTLR
__C.gbdt = edict()
__C.gbdt.learning_rate = 0.01
__C.gbdt.max_depth = 3
__C.gbdt.n_estimators = 50
__C.gbdt.subsample = 0.8

# XGB
__C.xgb = edict()
__C.xgb.learning_rate = 0.01
__C.xgb.max_depth = 3
__C.xgb.boosting_type = 'gbdt'
__C.xgb.num_leaves = 120
__C.xgb.min_data_in_leaf = 100
__C.xgb.feature_fraction = 0.8
__C.xgb.bagging_fraction = 0.8
__C.xgb.bagging_freq = 5
__C.xgb.lambda_l1 = 0.4
__C.xgb.lambda_l2 = 0.5


# LGB
__C.lgb = edict()
__C.lgb.learning_rate = 0.1
__C.lgb.max_depth = 3
__C.lgb.boosting_type = 'gbdt'
__C.lgb.num_leaves = 120
__C.lgb.min_data_in_leaf = 100
__C.lgb.feature_fraction = 0.8
__C.lgb.bagging_fraction = 0.8
__C.lgb.bagging_freq = 5
__C.lgb.lambda_l1 = 0.4
__C.lgb.lambda_l2 = 0.5


#DL



###
# 内部方法，实现yaml配置文件到dict的合并
def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v
# 自动加载yaml文件
def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r', encoding='utf-8') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


class Config():
    JIEBA_FLAG = True
    DEL_STOPWORD = False

    # 这个给定特征可以应用在 tf-idf 矩阵中，用以描述单词在文档中的最高出现率。假设一个词（term）在 80% 的文档中都出现过了，那它也许（在剧情简介的语境里）只携带非常少信息。
    MAX_DF = 0.8
    # 可以是一个整数（例如5）。意味着单词必须在 5 个以上的文档中出现才会被纳入考虑。设置为 0.2；即单词至少在 20% 的文档中出现 。
    MIN_DF = 0.2
    # 这个参数将用来观察一元模型（unigrams），二元模型（ bigrams） 和三元模型（trigrams）。参考n元模型（n-grams）。
    NGRAM_RANGE = 3 

if __name__ == '__main__':
    cfg_from_file('config.yml')
    print(cfg.emb.JIEBA_FLAG)
    print(cfg.emb.DEL_STOPWORD)
