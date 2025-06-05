import numpy as np
from copy import deepcopy


def aspect_oriented_tree(opt, token, head, as_start, as_end):
    """
    以句子 “the bread is top notch as well .” 为例。
    :param opt:
    :param token:       ['The', 'bread', 'is', 'top', 'notch', 'as', 'well', '.']
    :param head:        [2, 5, 5, 5, 0, 5, 6, 5]
    :param as_start:    1
    :param as_end:      2
    :return:
    """
    stoi = {}
    for i, t in enumerate(token):
        stoi[i] = t
    # stoi -> {0: "The", 1: 'bread', 2: 'is', 3: 'top', 4: 'notch', 5: 'as', 6: 'well', 7: '.'}

    children = []
    for _ in range(len(token)):
        children += [{}]
    # children -> [{}, {}, {}, {}, {}, {}, {}, {}]

    # 构建依存关系的子节点映射：
    # 遍历所有的 token，根据 head 列表中的信息，构建每个词的子节点关系。
    # children[i][j] = 1 和 children[j][i] = 1 表示在 children 中记录词语 i 和 j 之间的双向依赖关系（即它们在依存关系树中互为子节点和父节点）
    for i in range(len(token)):
        for j in range(len(head)):
            if head[j] - 1 == i and j not in children[i].keys() and head[j] != 0:
                children[i][j] = 1
                children[j][i] = 1

        if head[i] - 1 not in children[i].keys() and head[i] != 0:
            children[i][head[i] - 1] = 1
            children[head[i] - 1][i] = 1
    """
    children -> 
    [{1: 1},                            # 第0个单词跟第1个单词 存在依赖关系
     {0: 1, 4: 1},                      # 第1个单词跟第0,4个单词 存在依赖关系
     {4: 1},                            # 第2个单词跟第4个单词 存在依赖关系
     {4: 1},                            # 第3个单词跟第4个单词 存在依赖关系
     {1: 1, 2: 1, 3: 1, 5: 1, 7: 1},    # 第4个单词跟第1,2,3,5,7个单词存在依赖关系
     {4: 1, 6: 1},                      # 第5个单词跟第4,6个单词存在依赖关系
     {5: 1},                            # 第6个单词跟第5个单词存在依赖关系
     {4: 1}]                            # 第7个单词跟第4个单词存在依赖关系
    """

    # 为特定的方面词构建依存关系图：
    # 对于每个方面词（根据 as_start 和 as_end 范围），代码创建了一个 children_asp 的深拷贝，用来表示与该方面词相关的依存关系子树。
    # 初始化 head_idx 和 head_stack，分别记录该方面词的初始直接依赖关系（即直接连接的节点）和需要进一步探索的节点。
    children_asp_all = []
    for asp_idx in range(as_start, as_end):
        children_asp = deepcopy(children)
        head_idx = list(children_asp[asp_idx].keys())
        head_stack = deepcopy(head_idx)

        while (len(head_idx) < len(token)) and (len(head_stack) > 0):
            idx_in_sent = head_stack.pop(0)
            ids = list(children_asp[idx_in_sent].keys())

            for idx in ids:
                if idx not in head_idx and idx != asp_idx:
                    children_asp[asp_idx][idx] = children_asp[idx_in_sent][idx] + children_asp[asp_idx][idx_in_sent]
                    head_stack = [idx] + head_stack
                    head_idx += [idx]

        children_asp_all.append(children_asp)
    """
    children_asp_all -> 
    [[{1: 1},
      {0: 1, 4: 1, 2: 2, 3: 2, 5: 2, 7: 2, 6: 3},   # 扩展方面词与0,4,2,3,5,7,6单词有依赖关系,value值可以看作是对应的深度。
      {4: 1},
      {4: 1},
      {1: 1, 2: 1, 3: 1, 5: 1, 7: 1},
      {4: 1, 6: 1},
      {5: 1},
      {4: 1}]]
    """

    # distance based weighted matrix
    if 'bert' in opt.model_name:
        dm = np.ones((len(token), len(token))) * (np.inf)
    else:
        dm = np.ones((opt.max_length, opt.max_length)) * (np.inf)       # e.g., 10 x 10

    aspect_indices = list(range(as_start, as_end))
    for word_id in range(len(token)):
        distances = [np.inf]

        for child_id, asp_id in enumerate(aspect_indices):
            asp_child = children_asp_all[child_id][asp_id]
            try:
                distances.append(asp_child[word_id])
            except:
                distances.append(np.inf)
        real_distance = min(distances)

        for asp_id in aspect_indices:
            dm[asp_id][word_id] = real_distance
            dm[word_id][asp_id] = real_distance

    for asp_id in aspect_indices:
        for asp_mutual in aspect_indices:
            dm[asp_id][asp_mutual] = 1

    # self-loop
    for i in range(len(dm)):
        dm[i][i] = 1

    return dm

"""
array([[ 1.,  1., inf, inf, inf, inf, inf, inf, inf, inf],
       [ 1.,  1.,  2.,  2.,  1.,  2.,  3.,  2., inf, inf],
       [inf,  2.,  1., inf, inf, inf, inf, inf, inf, inf],
       [inf,  2., inf,  1., inf, inf, inf, inf, inf, inf],
       [inf,  1., inf, inf,  1., inf, inf, inf, inf, inf],
       [inf,  2., inf, inf, inf,  1., inf, inf, inf, inf],
       [inf,  3., inf, inf, inf, inf,  1., inf, inf, inf],
       [inf,  2., inf, inf, inf, inf, inf,  1., inf, inf],
       [inf, inf, inf, inf, inf, inf, inf, inf,  1., inf],
       [inf, inf, inf, inf, inf, inf, inf, inf, inf,  1.]])
"""


# 设置阈值，控制方面距离。
def aspect_oriented_tree_2(opt, token, head, as_start, as_end):
    stoi = {}
    for i, t in enumerate(token):
        stoi[i] = t

    children = []
    for _ in range(len(token)):
        children += [{}]

    for i in range(len(token)):
        for j in range(len(head)):
            if head[j] - 1 == i and j not in children[i].keys() and head[j] != 0:
                children[i][j] = 1
                children[j][1] = 1

        if head[i] - 1 not in children[i].keys() and head[i] != 0:
            children[i][head[i]-1] = 1
            children[head[i]-1][i] = 1

    children_asp_all = []
    for asp_idx in range(as_start, as_end):
        children_asp = deepcopy(children)
        head_idx = list(children_asp[asp_idx].keys())
        head_stack = deepcopy(head_idx)

        while (len(head_idx) < len(token)) and (len(head_stack) > 0):
            idx_in_sent = head_stack.pop(0)
            ids = list(children_asp[idx_in_sent].keys())

            for idx in ids:
                if idx not in head_idx and idx != asp_idx:
                    children_asp[asp_idx][idx] = children_asp[idx_in_sent][idx] + children_asp[asp_idx][idx_in_sent]
                    head_stack = [idx] + head_stack
                    head_idx += [idx]

        children_asp_all.append(children_asp)

    # 定义最大距离阈值
    max_distance = opt.sd
    # distance_matrices = []
    if 'bert' in opt.model_name:
        dm = np.ones((len(token), len(token))) * (np.inf)
    else:
        dm = np.ones((opt.max_length, opt.max_length)) * (np.inf)  # e.g., 10 x 10

    for distance_threshold in range(1, max_distance + 1):
        aspect_indices = list(range(as_start, as_end))
        for word_id in range(len(token)):
            distances = [np.inf]

            for child_id, asp_id in enumerate(aspect_indices):
                asp_child = children_asp_all[child_id][asp_id]
                try:
                    distance = asp_child[word_id]
                    if distance <= distance_threshold:
                        distances.append(distance)
                    else:
                        distances.append(np.inf)
                except:
                    distances.append(np.inf)
                real_distance = min(distances)

                for asp_id in aspect_indices:
                    dm[asp_id][word_id] = real_distance
                    dm[word_id][asp_id] = real_distance

            for asp_id in aspect_indices:
                for asp_mutual in aspect_indices:
                    dm[asp_id][asp_mutual] = 1

            # self-loop
            for i in range(len(dm)):
                dm[i][i] = 1

            # distance_matrices.append(dm)

    return dm


def dense_tree(opt, token, head):
    # dm = np.ones((len(token), len(token))) * np.inf
    if 'bert' in opt.model_name:
        dm = np.ones((len(token), len(token))) * (np.inf)
    else:
        dm = np.ones((opt.max_length, opt.max_length)) * (np.inf)

    # 构建依存关系的子节点映射
    children = []
    for _ in range(len(token)):
        children += [{}]

    # 构建依存关系的子节点映射
    for i in range(len(token)):
        for j in range(len(head)):
            if head[j] - 1 == i and j not in children[i].keys() and head[j] != 0:
                children[i][j] = 1
                children[j][i] = 1

        if head[i] - 1 not in children[i].keys() and head[i] != 0:
            children[i][head[i] - 1] = 1
            children[head[i] - 1][i] = 1

    # 根据依存关系更新距离矩阵
    for i in range(len(token)):
        for j in range(len(token)):
            if j in children[i].keys():
                dm[i][j] = 1
                dm[j][i] = 1

    # 自身距离设置为1
    for i in range(len(dm)):
        dm[i][i] = 1

    return dm


if __name__ == "__main__":
    # 示例输入
    class Opt:
        model_name = 'non_bert'
        max_length = 10

    opt = Opt()
    token = ['The', 'bread', 'is', 'top', 'notch', 'as', 'well', '.']
    head = [2, 5, 5, 5, 0, 5, 6, 5]
    as_start = 1
    as_end = 2

    # 调用函数
    # distance_matrices = aspect_oriented_tree_2(opt, token, head, as_start, as_end)
    distance_matrix = dense_tree(opt, token, head)

    # 打印不同距离阈值下的矩阵
    # for i, matrix in enumerate(distance_matrices):
    #     print(f"Distance threshold {i + 1}:")
    #     print(matrix)
    #     print()

    print(distance_matrix)
























