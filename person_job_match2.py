import json
import time

import numpy as np
import torch
from similarities import BertSimilarity

ordinal_scale = {
    '学历': ['文盲', '小学', '初中', '中专', '高中', '大专', '大学', '硕士', '博士'],
    '军衔': ['列兵', '上等兵', '下士', '中士', '上士', '四级军士长', '三级军士长', '二级军士长', '一级军士长',
             '少尉', '中尉', '上尉', '少校', '中校', '上校', '大校', '少将', '中将', '上将']
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return obj
        if isinstance(obj, np.floating):
            return int(obj * 10000) / 10000
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def check_constraint1(matching_matrix, pos_info, person_info, i, j):
    for constraint in pos_info[i]['强约束']:
        if constraint['type'] == 'nominal_1' and constraint['constraint'] != person_info[j][constraint['item']]:
            matching_matrix[i][j] = 0
        elif constraint['type'] == 'nominal_2':
            for x in constraint['constraint']:
                if x not in person_info[j][constraint['item']]:
                    matching_matrix[i][j] = 0
                    break
        elif constraint['type'] == 'ordinal':
            items = ordinal_scale[constraint['item']]
            if 'inf' in constraint and items.index(constraint['inf']) > items.index(person_info[j][constraint['item']]):
                matching_matrix[i][j] = 0
            if 'sup' in constraint and items.index(constraint['sup']) < items.index(person_info[j][constraint['item']]):
                matching_matrix[i][j] = 0
        elif constraint['type'] == 'interval' or constraint['type'] == 'ratio':
            if 'inf' in constraint and constraint['inf'] > person_info[j][constraint['item']]:
                matching_matrix[i][j] = 0
            if 'sup' in constraint and constraint['sup'] < person_info[j][constraint['item']]:
                matching_matrix[i][j] = 0
        else:
            pass


def check_constraint2(matching_matrix, pos_info, person_info, i, j):
    for constraint in pos_info[i]['弱约束']:
        if constraint['type'] == 'nominal_1' and constraint['constraint'] != person_info[j][constraint['item']]:
            continue
        elif constraint['type'] == 'nominal_2':
            for x in constraint['constraint']:
                if x not in person_info[j][constraint['item']]:
                    continue
        elif constraint['type'] == 'ordinal':
            items = ordinal_scale[constraint['item']]
            if 'inf' in constraint and items.index(constraint['inf']) > items.index(person_info[j][constraint['item']]):
                continue
            if 'sup' in constraint and items.index(constraint['sup']) < items.index(person_info[j][constraint['item']]):
                continue
        elif constraint['type'] == 'interval' or constraint['type'] == 'ratio':
            if 'inf' in constraint and constraint['inf'] > person_info[j][constraint['item']]:
                continue
            if 'sup' in constraint and constraint['sup'] < person_info[j][constraint['item']]:
                continue
        else:
            matching_matrix[i][j] = min(matching_matrix[i][j] * 1.01, 0.9999)


def check_historical_position(matching_matrix, pos_info, person_info, i, j):
    if pos_info[i]['战位名称'] in person_info[j]['历史战位']:
        matching_matrix[i][j] = 1


def compute_matching_matrix(model, pos_ids, pos_info, person_ids, person_info):
    task_description = [x['职责描述'] for x in pos_info]
    person_skills = [', '.join(x['专业技能']) for x in person_info]
    matching_matrix = model.similarity(task_description, person_skills)
    matching_matrix = matching_matrix.numpy()
    for i in range(len(pos_ids)):
        for j in range(len(person_info)):
            check_constraint1(matching_matrix, pos_info, person_info, i, j)
            check_constraint2(matching_matrix, pos_info, person_info, i, j)
            check_historical_position(matching_matrix, pos_info, person_info, i, j)
            matching_matrix[i][j] = int(matching_matrix[i][j] * 10000) / 10000
    return matching_matrix


# 舰员推荐
def recommend_person(model_path, person_info_path, position_info_path):
    pos_info = []
    pos_ids = []
    with open(position_info_path, encoding='utf-8') as f:
        content = json.load(f)
        for k, v in content.items():
            pos_ids.append(k)
            pos_info.append(v)
    person_ids = []
    person_info = []
    with open(person_info_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        for k, v in content.items():
            person_ids.append(k)
            person_info.append(v)
    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertSimilarity(model_name_or_path=model_path, device=model_device)
    matching_matrix = compute_matching_matrix(model, pos_ids, pos_info, person_ids, person_info)
    scores = matching_matrix[0]
    data = [[person_id, score] for person_id, score in zip(person_ids, scores)]
    data = sorted(data, key=lambda x: x[1], reverse=True)
    data = [{'person_id': x1, 'score': x2} for x1, x2 in data]
    res = {'code': 1, 'message': "舰员推荐结果获取成功", 'data': data}
    return res


# 战位推荐
def recommend_position(model_path, person_info_path, position_info_path):
    pos_info = []
    pos_ids = []
    with open(position_info_path, encoding='utf-8') as f:
        content = json.load(f)
        for k, v in content.items():
            pos_ids.append(k)
            pos_info.append(v)
    person_ids = []
    person_info = []
    with open(person_info_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        for k, v in content.items():
            person_ids.append(k)
            person_info.append(v)
    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertSimilarity(model_name_or_path=model_path, device=model_device)
    matching_matrix = compute_matching_matrix(model, pos_ids, pos_info, person_ids, person_info)
    scores = [x[0] for x in matching_matrix]
    data = [[pos_id, score] for pos_id, score in zip(pos_ids, scores)]
    data = sorted(data, key=lambda x: x[1], reverse=True)
    data = [{'position_id': x1, 'score': x2} for x1, x2 in data]
    res = {'code': 1, 'message': "战位推荐结果获取成功", 'data': data}
    return res


# 匹配矩阵生成
def recommend_matrix(model_path, person_info_path, position_info_path):
    pos_info = []
    pos_ids = []
    with open(position_info_path, encoding='utf-8') as f:
        content = json.load(f)
        for k, v in content.items():
            pos_ids.append(k)
            pos_info.append(v)
    person_ids = []
    person_info = []
    with open(person_info_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
        for k, v in content.items():
            person_ids.append(k)
            person_info.append(v)
    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertSimilarity(model_name_or_path=model_path, device=model_device)
    matching_matrix = compute_matching_matrix(model, pos_ids, pos_info, person_ids, person_info)
    data = {'header': person_ids, 'body': []}
    for i in range(len(matching_matrix)):
        data['body'].append({'position_id': pos_ids[i]})
        for j in range(len(matching_matrix[i])):
            data['body'][-1][person_ids[j]] = matching_matrix[i][j]
    res = {'code': 1, 'message': "匹配矩阵获取成功", 'data': data}
    return res


if __name__ == '__main__':
    start_time = time.time()
    res1 = recommend_person(model_path=r"C:\Users\zhena\Desktop\HJPROJECT\PyProject\bert-base-chinese",
                            person_info_path='test_data/all_data/persons.json',
                            position_info_path='test_data/position_data/position_35.json')
    json.dump(res1, open('result1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2, cls=NpEncoder)

    res2 = recommend_position(model_path=r"C:\Users\zhena\Desktop\HJPROJECT\PyProject\bert-base-chinese",
                              person_info_path='test_data/person_data/person_13.json',
                              position_info_path='test_data/all_data/positions.json')
    json.dump(res2, open('result2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2, cls=NpEncoder)

    res3 = recommend_matrix(model_path=r"C:\Users\zhena\Desktop\HJPROJECT\PyProject\bert-base-chinese",
                            person_info_path='test_data/all_data/persons.json',
                            position_info_path='test_data/all_data/positions.json')
    json.dump(res3, open('result3.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2, cls=NpEncoder)
    total_time = time.time() - start_time
    print('total running time: %.2f' % total_time)
