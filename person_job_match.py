import json
import time

import pandas as pd

import step2_bm25
import step3_rank
import step4_ensemble
from transformers import BertTokenizer
import numpy as np
from progress.bar import Bar
from similarities import BertSimilarity
import torch
from scipy.optimize import linear_sum_assignment

MAX_BM25 = 30
MAX_RANK = 10
MAX_ENSEMBLE = 5


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def pipeline(bm25_model, rank_model, person_list, tokenizer, job):
    bm25_result = step2_bm25.query(bm25_model, job, person_list, max_count=MAX_BM25)
    rank_result, num = step3_rank.query(rank_model, tokenizer, bm25_result, max_count=MAX_RANK)
    final_result = step4_ensemble.ensemble(rank_result, max_count=MAX_ENSEMBLE)
    return final_result, num


def match_model(person_info_path, job_info_path):
    bm25_model, person_list = step2_bm25.load_corpus(path=person_info_path)
    rank_model = step3_rank.load_model()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    jobs = json.load(open(job_info_path, 'r', encoding='utf-8'))
    jobs = jobs[:2]
    result_simple = {}
    result_all = {}
    bar = Bar("running", fill='#', max=100, suffix='%(percent)d%%')
    for job in bar.iter(jobs):
        result, num = pipeline(bm25_model, rank_model, person_list, tokenizer, job)
        job_desc = job['岗位描述']
        result_simple[job_desc] = {}
        for key in result:
            result_simple[job_desc][key] = []
            for candidate, score in result[key]:
                item = json.loads(candidate)
                info = [
                    f"姓名：{item['姓名']}，职位：{item['职位']}，工作科室：{item['工作部门']}，专业技能：{item['专业技能']}，得分：{score}，岗位编号：[{item['岗位编号']},{num}]",
                    score]
                result_simple[job_desc][key].append(info)
    json.dump(result_simple, open('result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2, cls=NpEncoder)


if __name__ == '__main__':
    start_time = time.time()
    match_model('data/person_info.jsonl', 'data/job_information_augmented.json')
    total_time = time.time() - start_time
    print('total running time: %.2f' % total_time)
