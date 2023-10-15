import argparse
# from pyserini.index import lucene
# from pyserini.search import lucene
import os
from pathlib import Path
import subprocess
from evaluation.evaluation import get_EM_F1, get_convinced_rate
# import ast
import json
import gpt_mrc
# from collections import Counter
from tqdm import tqdm
# from DPR import *

class DPR_Config_Controller:
    # This class is used to modify the config file for DPR in a consistent way, 
    # as opposed to manually editing the config file
    # Used when indexing/searching new passages, introducing new queries, etc.
    def __init__(self, path, file_type, name_in_config):
        self.path = path
        # relative path
        self.rel_path = './' + path[path.find('DPR/downloads'):]
        # convert the relative path to a dot-separated path
        assert './DPR/downloads' in self.rel_path
        self.DPR_dot_rel_path = ".".join((self.rel_path)\
                                .replace('./DPR/downloads/', '')\
                                .replace('/', '.')
                                .split('.')) # remove extension
        if self.DPR_dot_rel_path.endswith('.tsv') or \
            self.DPR_dot_rel_path.endswith('.csv'):
            self.DPR_dot_rel_path = self.DPR_dot_rel_path[:-4]
        self.file_type = file_type
        self.name_in_config = name_in_config
        if name_in_config is None: # filename
            self.name_in_config = self.DPR_dot_rel_path.split('.')[-1]
        
        os.chdir('./DPR')
        if self._is_in_config_file():
            print("Path already in config file")
            return
        else:
            self._add_psg_to_config()
    
    def _is_in_config_file(self):
        # check if the path is already in the config file
        if self.file_type == "passage":
            config_path = Path("./conf/ctx_sources/default_sources.yaml")
        elif self.file_type == "query":
            config_path = Path("./conf/datasets/retriever_default.yaml")
        with open(config_path, 'r') as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if self.DPR_dot_rel_path in line:
                    print(self.DPR_dot_rel_path)
                    self.name_in_config = lines[index-2].split(':')[0]
                    print(self.name_in_config, "already in config file")
                    return True
        return False

    def _add_psg_to_config(self):
        if self.file_type == "passage":
            config_path = Path("./conf/ctx_sources/default_sources.yaml")
        elif self.file_type == "query":
            config_path = Path("./conf/datasets/retriever_default.yaml")
        with open(config_path, 'r+') as f:
            # go to the end of the file
            f.seek(0, os.SEEK_END)
            # start writing from the end of the file
            f.write("\n")
            f.write(self.name_in_config + ":\n")
            if self.file_type == "passage":
                f.write("  _target_: dpr.data.retriever_data.CsvCtxSrc\n")
            elif self.file_type == "query":
                f.write("  _target_: dpr.data.retriever_data.CsvQASrc\n")
            else: 
                raise ValueError("file_type must be either passage or query")
            # f.write("  _target_: dpr.data.retriever_data.CsvCtxSrc\n") # we only support csv for now
            f.write("  file: " + self.DPR_dot_rel_path + "\n")
            f.close()
        with open('./dpr/data/download_data.py', 'r+') as f:
            # config also needs to be added to download_data.py
            # find the `RESOURCE_MAP` variable and insert a new entry
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "RESOURCES_MAP" in line:
                    lines.insert(i+1, "    \"" + self.DPR_dot_rel_path + "\": {" + "\n" + \
                        "        \"s3_url\": \"\",\n" + \
                        "        \"original_ext\": \"" + "."+self.path.split('.')[-1] + "\",\n" + \
                        "        \"compressed\": True,\n" + \
                        "        \"desc\": \"autogen\"\n" + \
                        "    },\n" )

                    break
            f.seek(0)
            f.writelines(lines)
            f.close()

def index_bm25(args):
    # index the passages with BM25
    index_path = args.out_path
    psg_path = Path(args.psg_path)
    psg_parent_path = psg_path.parent
    threads = args.threads
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    os.chdir(index_path)
    subprocess.run(["python", "-m", "pyserini.index", "-collection", "JsonCollection", 
        "-generator", "DefaultLuceneDocumentGenerator", "-threads", str(threads), 
        "-input", psg_path, "-index", index_path, "-storePositions", "-storeDocvectors", 
        "-storeRaw"])
    os.chdir('../')
    print("BM25 index created at", index_path)

def index_dpr(args):
    # requires different arguments
    dpr = DPR_Config_Controller(args.psg_path, "passage", args.psg_name_in_config)

    if os.path.exists(args.out_path) and not args.overwrite:
        raise ValueError('Output path already exists. Please check again if you want to overwrite it; add your command with the --overwrite flag.')
    elif not args.overwrite:
        print("Creating output directory...")
        os.makedirs(args.out_path) 
    else:
        if not os.path.exists(args.out_path):
            print("Nothing to overwrite. Creating output directory...")
        else:
            print("Overwriting existing output directory...")
    for i in range(args.threads):
        fout = open(f'{args.out_path}/log-{i}.txt', 'w')
        ret_code = subprocess.Popen(['python',  'generate_dense_embeddings.py', 
        f'model_file={args.model_file}',                                
        f'ctx_src={dpr.name_in_config}', 
        f'shard_id={i}', 
        f'num_shards={args.threads}', 
        f'out_file={args.out_path}/{dpr.name_in_config}'
        # f'batch_size=16',
        ], 
        stdout=fout, 
        stderr=fout) # output prefixed by the name of the passage file, useful for specifing in retrieval
        fout.close()   
    print("DPR indexing; Please check the log files in out_path for details.") 

def retrieve_bm25(args):
    query_path = Path(args.query_path)
    with open(query_path, 'r') as f:
        qas = f.readlines()
        queries = [qa.split('\t')[0] for qa in qas]
        # check the format of the queries, if the first field is not id,
        # then we need to convert it to [id, query] format
        if type(queries[0].split('\t')[0]) == str:
            print("Generating a new query file with id field...")
            queries = ['\t'.join([str(i), query]) for i, query in enumerate(queries)]
            new_query_path = query_path.parent / Path("reformatted-" + query_path.name)
            with open(new_query_path, 'w') as fout:
                fout.writelines("\n".join(queries))
    # change directory to pyserini root directory
    os.chdir('./pyserini')
    os.system('python -m pyserini.search.lucene \
    --topics {} --index {} --output {} --hits {} --bm25'
    .format(new_query_path, args.index_path, args.out_path + '/output.txt', args.top_k))
    # change directory back to the original directory
    os.chdir('../')

def retrieve_dpr(args):
    assert args.index_path is not None and args.query_path is not None and args.out_path is not None
    dpr_query = DPR_Config_Controller(args.query_path, "query", args.query_name_in_config)
    index_paths = args.index_path.split(',')
    psg_names = []
    direct_index_paths = [] # paths directly to the index files, as opposed to input paths which requires recursive search
    if args.psg_name_in_config is not None:
        psg_names = args.psg_name_in_config.split(',')
    else:
        for index_path in index_paths:
            for dirpath, dirnames, filenames in os.walk(index_path):
                # find all passage indix files in index_path recursively
                # assumes that all indices for one file collection reside in the same directory
                # assumes only index files end with `_[0-9]+`, and all indices start with _0 index`
                psg_config_name_in_dir = set([name.rsplit('_', 1)[0] for name in filenames
                    if name.endswith('_0')]) # all indices start with _0 index
                if len(psg_config_name_in_dir) == 0:
                    continue

                print(f'Found {len(psg_config_name_in_dir)} index,                   {psg_config_name_in_dir} in dir {dirpath} (We refer to index of one file collection as one index, not one index file.)')
                assert len(psg_config_name_in_dir) == 1, \
                    f'Found more than one index in dir {dirpath}, please specify the index name in config file.'
                psg_names.append(psg_config_name_in_dir.pop())
                direct_index_paths.append(dirpath)

    # assert len(index_paths) == len(psg_names)
    # psg_names = [psg_name.replace('-', '_') for psg_name in psg_names]
    print(os.getcwd())
    try:
        subprocess.call(f'python ./dense_retriever.py \
            model_file={args.model_file} \
            qa_dataset={dpr_query.name_in_config} \
            ctx_datatsets={"[" + ",".join(psg_names) + "]"} \
            encoded_ctx_files={"[" + ",".join([index_path + "/" + psg_name + "*" for index_path, psg_name in zip(direct_index_paths, psg_names)]) + "]"} \
            out_file={args.out_path}/output.json \
            | tee {args.out_path}/result.log', shell=True)
    except Exception as e:
        print(e)
        raise ValueError('DPR retrieval failed.')

def voted(indiv_answers_list, questions, naive):
    # this function receives a list of lists of answers produced by ChatGPT doing ODQA
    # and returns a list of answers with voting, conducted implicitly by GPT-3.5 if not naive
    voted_answers = []
    
    for i in range(len(indiv_answers_list)):
        if not naive:
            # produce majority-voted answers through GPT-3.5
            response = gpt_mrc.gpt_vote(indiv_answers_list[i], questions[i])
            voted_answer = response['choices'][0]['text']
        else:
            # produce majority-voted answers through naive voting
            voted_answer = max((indiv_answers_list[i]), key=lambda x: (indiv_answers_list.count(x), -indiv_answers_list.index(x)))
        voted_answers.append(voted_answer)
    return voted_answers

def read_FiD(args):
    assert args.retrieve_result_path is not None and args.out_path is not None
    if not os.getcwd().endswith('FiD-main'):
        os.chdir('./FiD-main')
    # use run here instead of call, so that we wait for the subprocess to finish
    if args.sample:
        with open(f'{args.retrieve_result_path}', 'r') as fin:
            with open(f'{args.retrieve_result_path}.sample.json', 'w') as fout:
                in_data = json.load(fin)
                out_data = in_data[-300:]
                json.dump(out_data, fout, indent=4)
    args.retrieve_result_path = f'{args.retrieve_result_path}.sample.json' if args.sample else args.retrieve_result_path
    
    if not args.vote:
        subprocess.run(f"python test_reader.py \
        --model_path {args.reader_model} \
        --eval_data {args.retrieve_result_path} \
        --per_gpu_batch_size 1 \
        --n_context {args.top_k} \
        --name my_test \
        --checkpoint_dir {args.out_path} \
        --write_results \
        ", shell=True)
    else:
        # with voting, we need to split the retrieval results 
        # into 10 parts and dump them into 10 files
        # and run 10 times of FiD reader
        with open(args.retrieve_result_path, 'r') as f:
            qas = json.load(f)
            out_list = [[] for _ in range(10)]
            for qac in qas:
                for i in range(10):
                    newcell = {}
                    newcell['question'] = qac['question']
                    newcell['answers'] = qac['answers']
                    newcell['ctxs'] = qac['ctxs'][i*10:(i+1)*10]
                    out_list[i].append(newcell)
            
            for i in range(10):
                with open(f'{args.retrieve_result_path}-{i}.json', 'w') as fout:
                    json.dump(out_list[i], fout, indent=4)
            
        print('Start running FiD reader with voting...')
        for i in tqdm(range(10)):
            subprocess.run(f"python test_reader.py \
            --model_path {args.reader_model} \
            --eval_data {args.retrieve_result_path}-{i}.json \
            --per_gpu_batch_size 1 \
            --n_context 10 \
            --name my_test_{i} \
            --checkpoint_dir {args.out_path} \
            --write_results \
            ", shell=True)

        # merge the results
        merged_dir = f'{args.retrieve_result_path}/FiD-top100-NQ/my_test'
        os.makedirs(merged_dir, exist_ok=False)
        with open(merged_dir + '/final_output.txt', 'w') as f_out:
            indiv_output = [[] for _ in range(10)]
            for i in range(10):
                with open(args.out_path + f'/my_test_{i}/final_output.txt', 'r') as f_in:
                    indiv_output[i] = f_in.readlines()
            
            voted_output = []
            for i in range(len(indiv_output[0])):
                # output the most frequent answer, if there is a tie, output the first one in the original list
                indiv_answers = [indiv_output[j][i].strip() for j in range(10)]
                voted_answer = max(set(indiv_answers), key=indiv_answers.count)
                voted_output.append(voted_answer)
                f_out.write(voted_answer + '\n')

    # f1-score integrated in FiD test_reader.py

def read_GPT(args, prompt_idx=0):
    assert args.retrieve_result_path is not None and args.out_path is not None
    # TODO: top_k hard coded
    if not args.vote:
        output = gpt_mrc.read_dpr_output(args.retrieve_result_path, int(args.top_k), args.multi_answer, args.disinfo, args.extract_and_read, args.size_limit, args.holdback, args.sample, prompt_idx)
    else:
        output = gpt_mrc.multi_reader_vote(args.retrieve_result_path, int(args.top_k), args.multi_answer, args.disinfo, args.size_limit, args.holdback, args.sample,)
        with open(args.out_path / 'before-vote.txt', 'w') as f:
            for i in output:
                f.write(str(i) + '\n')

        # get questions from retrieve_result_path
        questions = []
        with open(args.retrieve_result_path, 'r') as f:
            qas = json.load(f)
            for qa in qas:
                questions.append(qa['question'])
        output = voted(output, questions, naive=args.naive_vote)
    return output
    # for i in output.keys():
    #     os.makedirs(args.out_path + '/' + f'GPT3_5-{i}ctxs.txt', exist_ok=True)
    #     with open(args.out_path + '/' + f'GPT3_5-{i}ctxs.txt', 'w') as f:
    #         for j in output[i]:
    #             f.write(j + '\n')

def load_dpr(args):
    if os.path.exists(args.retrieve_result_dir / 'result.json'):
        # DPR result
        with open(args.retrieve_result_dir / 'result.json', 'r') as f:
            dpr_style_result = json.load(f)
    elif os.path.exists(args.retrieve_result_dir / 'output.json'):
        # BM25 converted result
        with open(args.retrieve_result_dir / 'output.json', 'r') as f:
            dpr_style_result = json.load(f)
    else:
        raise ValueError(f"Cannot find result.json or output.json in {args.retrieve_result_dir}")

    if args.sample:
        dpr_style_result = dpr_style_result[-300:]
    return dpr_style_result

def evaluate(args):
    # summarize all needed metric numbers into one json file
    assert args.retrieve_result_dir is not None and args.read_result_path is not None and args.out_path is not None
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        print(f"Created directory {args.out_path}")
    out_metric = {}
    
    # get retrieval data from retrieve_result_path
    metrics = {'accuracy': sampling_points, 'genRate': sampling_points, 'poisonedRate': sampling_points,  'convincedRate': [], 'EM&F1': []}
    dpr_style_result = load_dpr(args)

    for metric in metrics:
        if metric == 'accuracy':
            for k in metrics[metric]:
                hit_num = 0
                for item in dpr_style_result:
                    hit_num += any([ctx['has_answer'] for ctx in item['ctxs'][:k]])
                out_metric[f'accuracy@{k}'] = hit_num / len(dpr_style_result)
        
        elif metric == 'genRate':
            # percentage of generated passages in top-k retrieved passages
            if os.path.exists(args.retrieve_result_dir / 'output.json'):
                # with open(args.retrieve_result_dir / 'output.json', 'r') as f:
                #     retrieve_result = json.load(f)
                for k in metrics[metric]:
                    num_gen = len([ctxs for q in dpr_style_result for ctxs in q['ctxs'][:k] if ctxs['title'].startswith('<G>')])
                    num_total = len([ctxs for q in dpr_style_result for ctxs in q['ctxs'][:k]])
                    out_metric[f'genRate@{k}'] = round(num_gen / num_total, 4)
            elif os.path.exists(args.retrieve_result_dir / 'result.json'):
                # with open(args.retrieve_result_dir / 'result.json', 'r') as f:
                #     dpr_style_result = json.load(f)
                    for k in metrics[metric]:
                        gen_num = 0
                        total_num = 0
                        for item in dpr_style_result:
                            gen_num += len([ctx for ctx in item['ctxs'][:k] if not ctx['docid'].startswith('base')])
                            total_num += len(item['ctxs'][:k])
                        out_metric[f'genRate@{k}'] = round(gen_num / total_num, 4)
            else:
                raise ValueError(f"Cannot find result.json or output.json in {args.retrieve_result_dir}")
        
        elif metric == 'poisonedRate':
            # percentage of questions that have at least one poisoned(generated) passage in top-k retrieved passages
            if os.path.exists(args.retrieve_result_dir / 'output.json'):
                # with open(args.retrieve_result_dir / 'output.json', 'r') as f:
                #     retrieve_result = json.load(f)
                for k in metrics[metric]:
                    num_poisoned = len([q for q in dpr_style_result if any([ctxs['title'].startswith('<G>') for ctxs in q['ctxs'][:k]])])
                    out_metric[f'poisonedRate@{k}'] = round(num_poisoned / len(dpr_style_result), 4)
            elif os.path.exists(args.retrieve_result_dir / 'result.json'):
                # with open(args.retrieve_result_dir / 'result.json', 'r') as f:
                #     dpr_style_result = json.load(f)
                    for k in metrics[metric]:
                        poisoned_num = 0
                        for item in dpr_style_result:
                            poisoned_num += any([ctx for ctx in item['ctxs'][:k] if not ctx['docid'].startswith('base')])
                        out_metric[f'poisonedRate@{k}'] = round(poisoned_num / len(dpr_style_result), 4)
            else:
                raise ValueError(f"Cannot find result.json or output.json in {args.retrieve_result_dir}")
            
        elif metric == 'EM&F1':
            get_EM_F1(dpr_style_result, args.read_result_path, out_metric)
        
        # elif metric == 'convincedRate':
            # get_convinced_rate(args.read_result_path, out_metric, args.sample)

    with open(args.out_path / 'metrics.json', 'w') as f:
        json.dump(out_metric, f)
    print("Metrics: ",  out_metric)
    print(f"Saved metrics to {args.out_path / 'metrics.json'}")
    return out_metric

if __name__ == "__main__": 
    sampling_points = [3, 5, 10, 20, 50, 100] # six settings for reader evaluation

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", choices=['index', 'retrieve', 
        'read', 'evaluate', 'multi-read-evaluate'], required=True)
    parser.add_argument('--retriever', choices=['BM25', 'DPR'], 
        )
    parser.add_argument('--reader', choices=['FLAN-T5', 'FiD', 'GPT'], 
        )
    parser.add_argument('--psg_path', type=str)
    parser.add_argument('--out_path', type=str, help='dir to the output')
    parser.add_argument('--index_path', type=str, help='path to the index \
        directory, could be multiple directories separated by comma')
    parser.add_argument('--model_file', type=str, help='path to the DPR model file')
    parser.add_argument('--query_path', type=str)
    parser.add_argument('--psg_name_in_config', type=str, help='name of the \
        passage file in the config file. could be multiple names separated by comma \
        if unspecified, use the filename', default=None)
    parser.add_argument('--query_name_in_config', type=str, help='name of the \
        query file in the config file. if unspecified, use the filename', default=None)
    parser.add_argument('--threads', type=int, default=1, help='number of \
        threads to use for indexing')
    parser.add_argument('--retrieve_result_path', type=str)
    parser.add_argument('--overwrite', action='store_true', help='overwrite \
        existing index')
    parser.add_argument('--top_k', type=int, help='number of \
        passages to retrieve/number of retrieved passages to read', default=100)
    parser.add_argument('--read_result_path', type=str, help='path to the \
                        output file of the reading task')
    parser.add_argument('--retrieve_result_dir', type=str, help='dir to the \
                        output files (json result and log file) of the retrieval task')
    parser.add_argument('--reader_model', type=str, help='path to the model file')
    parser.add_argument('--multi_answer', action='store_true', help='whether \
                        to reveal to readers that the answer to produce list of answers')
    parser.add_argument('--disinfo', action='store_true', help='whether \
                        to reveal to readers that the passages contain disinformation')
    parser.add_argument('--extract_and_read', action='store_true', help='whether \
                        to extract passages from the retrieved passages and read them')
    parser.add_argument('--size_limit', type=int, help='size limit of the \
                        input questions to the reader')
    parser.add_argument('--vote', action='store_true', help='whether to use \
                        voting to aggregate the answers')
    parser.add_argument('--holdback', action='store_true', help='whether to \
                        give instructions to holdback answering in the prompts')
    parser.add_argument('--sample', action='store_true', help='whether to \
                        sample the questions to do QA, instead of reading all passages. Here \
                        we use the last 300 questions as the sample.')
    parser.add_argument('--multiprompt', action='store_true', help='whether to \
                        use multiple prompts to read the passages')
    parser.add_argument('--naive_vote', action='store_true', help='whether to  \
                        use naive voting to aggregate the answers')
    
    args = parser.parse_args()

    if args.task == 'index':
        assert args.psg_path is not None and args.out_path is not None
        psg_path = Path(args.psg_path)
        if not psg_path.exists():
            raise ValueError('Path to passage file does not exist.')
        else:
            # make the path its parent directory, as per pyserini requirements
            parent_psg_path = psg_path.parent
        if args.retriever == 'BM25':
            index_bm25(args)
        if args.retriever == 'DPR':
            index_dpr(args)

    elif args.task == 'retrieve':
        assert args.index_path is not None and \
            args.query_path is not None and args.out_path is not None
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)
        if args.retriever == 'BM25':
            retrieve_bm25(args)
        if args.retriever == 'DPR':
            retrieve_dpr(args)

    elif args.task == 'read':
        if args.reader == 'FiD':
            read_FiD(args)
        elif args.reader == 'GPT':
            read_GPT(args)

    elif args.task == 'evaluate':
        args.retrieve_result_dir = Path(args.retrieve_result_path).parent
        args.out_path = Path(args.out_path)
        evaluate(args)

    elif args.task == 'multi-read-evaluate':
        EM_scores = {}
        F1_scores = {}
        convinvedRate_scores = {}
        if args.reader == 'FiD':
            args.out_path = Path(args.retrieve_result_path).parent
            parent_result_out_path = Path(args.out_path)
            # for top_k in sampling_points:
            for top_k in [100]:
                print(f"Using top {top_k} passages for reading")
                args.top_k = top_k
                args.out_path = parent_result_out_path / f'FiD-top{top_k}-NQ'
                if not os.path.exists(args.out_path):
                    read_FiD(args)
                    print(f"Reading done, saved to {args.out_path}")
                args.read_result_path = args.out_path / 'my_test' / 'final_output.txt'
                args.retrieve_result_dir = Path(args.retrieve_result_path).parent
                metrics = evaluate(args)
                EM_scores[top_k] = metrics['EM']
                F1_scores[top_k] = metrics['F1']
                convinvedRate_scores[top_k] = metrics['convincedRate']

            print(f"EM scores: {EM_scores}")
            print(f"F1 scores: {F1_scores}")
        
        elif args.reader == 'GPT':
            args.out_path = Path(args.retrieve_result_path).parent
            parent_result_out_path = Path(args.out_path)
            # for top_k in [0, 3, 5, 10]:
            for top_k in [args.top_k]:
                print(f"Using top {top_k} passages for reading")
                args.top_k = top_k
                args.out_path = parent_result_out_path / f'GPT-top{top_k}-NQ'
                if args.multiprompt:
                    for i in range(1, 6):
                        print(f"Reading with prompt {i}")
                        args.out_path = parent_result_out_path / f'GPT-top{top_k}-NQ' / f'prompt{i}'
                        if not os.path.exists(args.out_path):
                            os.makedirs(args.out_path / 'my_test')
                        output = read_GPT(args, i)
                        with open(args.out_path / 'my_test' / 'final_output.txt', 'w') as f:
                            f.write('\n'.join(output))  
                        print(f"Reading done, saved to {args.out_path}")
                        args.read_result_path = args.out_path / 'my_test' / 'final_output.txt'
                        args.retrieve_result_dir = Path(args.retrieve_result_path).parent
                        metrics = evaluate(args)
                        EM_scores[top_k] = metrics['EM']
                        F1_scores[top_k] = metrics['F1'] 
                        print(f"EM scores: {EM_scores}")
                        print(f"F1 scores: {F1_scores}")               
                elif not os.path.exists(args.out_path):
                    os.makedirs(args.out_path / 'my_test')
                    output = read_GPT(args)
                    with open(args.out_path / 'my_test' / 'final_output.txt', 'w') as f:
                        f.write('\n'.join(output))
                    print(f"Reading done, saved to {args.out_path}")
                    args.read_result_path = args.out_path / 'my_test' / 'final_output.txt'
                    args.retrieve_result_dir = Path(args.retrieve_result_path).parent
                    metrics = evaluate(args)
                    EM_scores[top_k] = metrics['EM']
                    F1_scores[top_k] = metrics['F1']
                    print(f"EM scores: {EM_scores}")
                    print(f"F1 scores: {F1_scores}")
                else:
                    args.read_result_path = args.out_path / 'my_test' / 'final_output.txt'
                    args.retrieve_result_dir = Path(args.retrieve_result_path).parent
                    
                    metrics = evaluate(args)
                    EM_scores[top_k] = metrics['EM']
                    F1_scores[top_k] = metrics['F1']
                    print(f"EM scores: {EM_scores}")
                    print(f"F1 scores: {F1_scores}")
        # print(f"EM scores: {EM_scores}")
        # print(f"F1 scores: {F1_scores}")
        # print(f"convincedRate scores: {convinvedRate_scores}")
        
        with open(parent_result_out_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f)
