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

if __name__ == '__main__':
    query_file = '/mnt/victoria/data3/liangming/yikangpan/nela-gt-main/question-generation/attempt4-large/politics/reliable/qas/politics-QG2-num-203-T0.json'
    result_file = '/mnt/victoria/data3/liangming/yikangpan/nela-gt-main/embeddings-DPR/definitive-large/politics/DPR-malicious-1psg/output.json'
    print("Gold@k scores: ")
    print("k=1 ", gold_at_k(result_file, query_file, 1))
    print("k=5, ", gold_at_k(result_file, query_file, 5))
    print("k=10, ", gold_at_k(result_file, query_file, 10))
    print("k=20, ", gold_at_k(result_file, query_file, 20))
    print("k=100, ", gold_at_k(result_file, query_file, 100))
    print("Gen@k scores:")
    print("k=1, " , gen_at_k(result_file, 1))
    print("k=5, " , gen_at_k(result_file, 5))
    print("k=10, " , gen_at_k(result_file, 10))
    print("k=20, " , gen_at_k(result_file, 20))
    print("k=100, " , gen_at_k(result_file, 100))