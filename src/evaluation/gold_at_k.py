# this function calculates the gold@k score
# Given a DPR result and its original query file with its reference passages

import json

def gold_at_k(result_file, query_file, k=100):
    with open(result_file, 'r', encoding='utf-8') as f:
        result = json.load(f)
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    total = 0
    correct = 0
    for idx, q in enumerate(queries):
        total += 1
        # print(q)
        q_title = q['title']
        if q_title in [i['title'] for i in result[idx]['ctxs'][:k]]:
            correct += 1
    return correct/total

def gen_at_k(result_file, k=100):
    with open(result_file, 'r', encoding='utf-8') as f:
        result = json.load(f)
    total = 0
    correct = 0
    for idx, res in enumerate(result):
        for ctx in res['ctxs'][:k]:
            total += 1
            title = ctx['title']
            if title.startswith('<G>'):
                correct += 1
    return correct/total
