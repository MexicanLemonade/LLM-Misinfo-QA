"""
Adapted from:
https://github.com/facebookresearch/FiD/blob/main/src/evaluation.py
Dependencies:
https://github.com/facebookresearch/FiD#dependencies
"""

import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter
from rouge import Rouge
from pathlib import Path
import os


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for _, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1(prediction, ground_truths):
    return max([f1_score(prediction, gt) for gt in ground_truths])


def rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def rl(prediction, ground_truths):
    return max([rougel_score(prediction, gt) for gt in ground_truths])


## file-level evaluation ... ### 
def eval_recall(infile):

    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    has_answer_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = ' || '.join(line['output'])

        if has_answer(answer, output, tokenizer):
            has_answer_count += 1

        answer_lengths.append(len(output.split()))

    recall = round(has_answer_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return recall, lens


def eval_question_answering(infile):

    lines = open(infile, 'r').readlines()[0:]

    exact_match_count = 0
    answer_lengths = []
    f1_scores = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output']
        # print(output)

        # if ems(output, answer): # EM evaluation
        #     exact_match_count += 1

        if any([ems(o, answer) for o in output]): # EM evaluation
            exact_match_count += 1
        
        answer_lengths.append(sum([len(o.split()) for o in output])/len(output))

        f1_scores.append(max([f1(o, answer) for o in output])) # F1 evaluation

    em = round(exact_match_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)
    f1_score = round(np.mean(f1_scores), 4)

    return em, lens, f1_score


def eval_fact_checking(infile):

    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    exact_match_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        if answer == ["refutes"]:
            answer = ["refutes", "no", "false"]
        if answer == ["supports"]:
            answer = ["supports", "yes", "true"]

        if has_answer(answer, output, tokenizer):
            exact_match_count += 1
        
        answer_lengths.append(len(output.split()))

    em = round(exact_match_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return em, lens


def eval_dialogue_system(infile):

    lines = open(infile, 'r').readlines()[1:]

    f1_scores = []
    rl_scores = []
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        f1_scores.append(f1(output, answer))
        rl_scores.append(rl(output, answer))
        answer_lengths.append(len(output.split()))

    F1 = round(np.mean(f1_scores), 4)
    RL = round(np.mean(rl_scores), 4)
    lens = round(np.mean(answer_lengths), 4)

    return F1, RL, lens

def get_EM_F1(retrieve_result, read_result_path, out_metric):
    # create a json result file from retrieval and read results for evaluation
    read_result_path = Path(read_result_path)
    eval_file_path = read_result_path.parent /'for-eval.jsonl'
    with open(eval_file_path, 'w') as f:
        # if os.path.exists(retrieve_result_dir / 'output.json'):
        #     result_json = retrieve_result_dir / 'output.json'
        # elif os.path.exists(retrieve_result_dir / 'result.json'):
        #     result_json = retrieve_result_dir / 'result.json'
        # else:
        #     raise ValueError(f'No output.json or result.json found in {retrieve_result_dir}')
        # retrieve_result = json.load(open(result_json, 'r'))
        read_result = open(read_result_path, 'r').readlines()
        for q, a in zip(retrieve_result, read_result):
            if '\t' not in a:
                predicted = [a.strip()]
            else:
                predicted = [sing.strip() for sing in a.split('\t')]
            out_dict = {'answer': q['answers'], # ref answer \
                        'output': predicted # predicted answer \
                        }
            f.write(json.dumps(out_dict) + '\n')

    em, lens, f1 = eval_question_answering(eval_file_path)
    out_metric['EM'] = em
    out_metric['F1'] = f1
    out_metric['lens'] = lens
    # return out_metric

def get_convinced_rate(read_result_path, out_metric, sample, false_answer_path):
    # create a json result file from retrieval and read results for evaluation
    read_result_path = Path(read_result_path)
    eval_file_path = read_result_path.parent /'for-eval-false-answer.jsonl'
    with open(eval_file_path, 'w') as f:
        # false_answer_path = 
        with open(false_answer_path, 'r') as f2:
            false_result = f2.readlines()
        read_result = open(read_result_path, 'r').readlines()
        for false_a, a in zip(false_result, read_result):
            if '\t' not in a:
                predicted = a.strip()
            else:
                predicted = a.split('\t')[1].strip()
            out_dict = {'answer': [false_a], # ref answer \
                        'output':[predicted] # predicted answer \
                        }
            f.write(json.dumps(out_dict) + '\n')

    em, _, _ = eval_question_answering(eval_file_path)
    out_metric['convincedRate'] = em
    # out_metric['lens'] = lens
    # return out_metric