import json
import copy
import random

random.seed(10)

f1 = open('person_info.jsonl', 'r', encoding='utf-8')
person_info = {}
for i, line in enumerate(f1):
    line = json.loads(line)
    new_dict = copy.deepcopy(line)
    new_dict.pop('人员编号')
    new_dict.pop('岗位编号')
    new_dict['性别'] = '男' if random.random() <= 0.5 else '女'
    new_dict['婚姻状况'] = random.choice(['未婚'] * 5 + ['已婚'] * 5 + ['离异'])
    new_dict['民族'] = random.choice(['汉族'] * 10 + ['满族'] * 2 + ['壮族'])
    new_dict['军衔'] = random.choice(['少校', '中校', '上校'])
    new_dict['语言能力'] = random.choice([['汉语']] * 10 + [['汉语', '英语']] * 5 + [['汉语', '英语', '法语']])
    new_dict['400m成绩'] = random.randint(55, 80)
    new_dict['引体向上成绩'] = random.randint(10, 30)
    new_dict['双眼矫正视力'] = random.randint(6, 12) / 10
    new_dict['历史战位'] = [new_dict['职位']] if i % 11 == 3 else []
    # person_info = {}
    person_info[('person_id_' + str(line['人员编号']))] = new_dict
    # json.dump(person_info, open('person_data/person_' + str(i) + '.json', 'w', encoding='utf-8'),
    #           ensure_ascii=False, indent=2)
json.dump(person_info, open('persons.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

f2 = open('job_information_augmented.json', 'r', encoding='utf-8')
jobs = json.load(f2)
job_info = {}
for i, job in enumerate(jobs):
    constraint1 = []
    constraint2 = []
    if job['岗位'] == "船长":
        constraint1.append({'item': '军衔', 'type': 'ordinal', 'inf': '上校', })
    if job['岗位'] == "副船长":
        constraint1.append({'item': '军衔', 'type': 'ordinal', 'inf': '少校', 'sup': '上校'})
    if job['岗位'] == "航海系统技术员":
        constraint1.append({'item': '语言能力', 'type': 'nominal_2', 'constraint': ['汉语', '英语']})
    if job['岗位'] == "通信人员":
        constraint1.append({'item': '语言能力', 'type': 'nominal_2', 'constraint': ['汉语', '英语', '法语']})
    if job['岗位'] == "主炮火控人员":
        constraint1.append({'item': '身高', 'type': 'ratio', 'inf': 170, 'sup': 185})
        constraint1.append({'item': '性别', 'type': 'nominal_1', 'constraint': '男'})
    if job['岗位'] == "航海观察员":
        constraint1.append({'item': '双眼矫正视力', 'type': 'ratio', 'inf': 1.2})

    if job['岗位'] == "油料人员":
        constraint2.append({'item': '身高', 'type': 'ratio', 'inf': 150, 'sup': 170})
    if job['岗位'] == "深弹人员":
        constraint2.append({'item': '身高', 'type': 'ratio', 'inf': 170, 'sup': 180})
    if job['岗位'] == "政治教导员":
        constraint2.append({'item': '军衔', 'type': 'ordinal', 'inf': '少校'})
    # job_info = {}
    job_info[('position_id_' + str(job['岗位编号']))] = {'部门': job['部门'], '战位名称': job['岗位'],
                                                         '职责描述': job['岗位描述'], '强约束': constraint1,
                                                         '弱约束': constraint2}
    # json.dump(job_info, open('position_data/position_' + str(i) + '.json', 'w', encoding='utf-8'),
    #           ensure_ascii=False, indent=2)
json.dump(job_info, open('positions.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
