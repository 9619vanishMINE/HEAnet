from datasets import load_ft
from utils import hypergraph_utils as hgut


def load_feature_construct_H(data_dir,
                             m_prob=1,
                             K_neigs=[10],
                             is_probH=True,
                             split_diff_scale=False,
                             use_mvcnn_feature=False,
                             use_gvcnn_feature=True,
                             use_mvcnn_feature_for_structure=False,
                             use_gvcnn_feature_for_structure=True):
    """

    :param data_dir: directory of feature data
    :param m_prob: parameter in hypergraph incidence matrix construction
    :param K_neigs: the number of neighbor expansion
    :param is_probH: probability Vertex-Edge matrix or binary
    :param use_mvcnn_feature:
    :param use_gvcnn_feature:
    :param use_mvcnn_feature_for_structure:
    :param use_gvcnn_feature_for_structure:
    :return:
    """

    # construct feature matrix;
    fts = None
    if use_mvcnn_feature:
        fts = hgut.feature_concat(fts, mvcnn_ft)
    if use_gvcnn_feature:
        fts = hgut.feature_concat(fts, gvcnn_ft)  # fts, GVCNN feature
    if fts is None:
        raise Exception(f'None feature used for model!')

    # construct hypergraph incidence matrix
    print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
    H = None
    if use_mvcnn_feature_for_structure:  # MVCNN feature build Hyperedges
        tmp = hgut.construct_H_with_KNN(mvcnn_ft, K_neigs=K_neigs,  # 求K_neigs个邻居距离矩阵
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob)
        H = hgut.hyperedge_concat(H, tmp)  # Hyperedges concat;
    if use_gvcnn_feature_for_structure:  # GVCNN feature build Hyperedges
        tmp = hgut.construct_H_with_KNN(gvcnn_ft, K_neigs=K_neigs,
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob)  # (12311, 12311)
        H = hgut.hyperedge_concat(H, tmp)  # (12311, 24622); (H>0).sum(axis=0)Hyperedges, contact 10 points
    if H is None:
        raise Exception('None feature to construct hypergraph incidence matrix!')
    # gvcnn_ft feature; label; 训练集索引; 测试集; 两个数据的最近n个邻居矩阵
    return fts, lbls, idx_train, idx_test, H
