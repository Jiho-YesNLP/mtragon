"""
Random sample of the retrieval results, compare documents w.r.t. query to assess quality.
"""
import code
import os
import json
import argparse
import random
from collections import defaultdict
from functools import partial

# input file: retrieved results
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the retrieved results file.')
args = arg_parser.parse_args()

if not os.path.exists(args.input_file):
    raise FileNotFoundError(f"Input file {args.input_file} does not exist.")

doc_ids = set()

# Retrieval results from the input jsonl file
retrieved = {}
with open(args.input_file, 'r') as f:
    for l in f:
        item = json.loads(l)
        topic_id, turn_id = item['task_id'].split('<::>')
        if topic_id not in retrieved:
            retrieved[topic_id] = {}
        retrieved[topic_id][int(turn_id)] = item['contexts']
        doc_ids.update([ctx['document_id'] for ctx in item['contexts']])

ret_data_dir = './data/raw/human/retrieval_tasks'
# Read in ground true retrieval results
qrels_files = {
    'clapnq': 'clapnq/qrels/dev.tsv',
    'govt': 'govt/qrels/dev.tsv',
    'fiqa': 'fiqa/qrels/dev.tsv',
    'cloud': 'cloud/qrels/dev.tsv',
}

qrels = {}
for dataset, qrels_file in qrels_files.items():
    with open(os.path.join(ret_data_dir, qrels_file), 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split('\t')
            query_id, doc_id, relevance = parts[0], parts[1], int(parts[2])
            doc_ids.add(doc_id)
            topic_id, turn_id = query_id.split('<::>')
            if topic_id not in qrels:
                qrels[topic_id] = defaultdict(list)
            qrels[topic_id][int(turn_id)].append(
                {'document_id': doc_id, 'score': relevance}
            )

# Collect all the passages either mentioned in GT or retrieved
passages = {}
for corpus in qrels_files.keys():
    corpus_file = f'./data/raw/corpora/passage_level/{corpus}.jsonl'
    with open(corpus_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            if item['id'] in doc_ids:
                passages[item['id']] = item

# Read questions
questions_files = {
    'clapnq': 'clapnq/clapnq_questions.jsonl',
    'govt': 'govt/govt_questions.jsonl',
    'fiqa': 'fiqa/fiqa_questions.jsonl',
    'cloud': 'cloud/cloud_questions.jsonl',
}

questions = {}
for dataset, questions_file in questions_files.items():
    with open(os.path.join(ret_data_dir, questions_file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            topic_id, turn_id = item['_id'].split('<::>')
            if topic_id not in questions:
                questions[topic_id] = {}
            questions[topic_id][int(turn_id)] = item['text']

def sample_topics(r, q, gt, passages):
    # sample one topic
    k = random.choice(list(r.keys()))
    print(f"Sampled topic: {k}")
    for turn in sorted(r[k].keys()):
        print(f"\nTurn {turn}: {q[k][turn]}")
        print("\n# GT passages:")
        for ctx in gt.get(k, {}).get(turn, [])[:3]:
            pid = ctx['document_id']
            score = ctx['score']
            print(f"  [Score: {score}] {pid}: {repr(passages[pid]['text'][:400])}...")
        print("\n#Retrieved passages:")
        for ctx in r[k][turn][:3]:
            pid = ctx['document_id']
            print(f"  [Score: {ctx.get('score', 'N/A')}] {pid}: {repr(passages[pid]['text'][:400])}...")
        # wait for user input to proceed to next turn
        input("\nPress Enter to continue to the next turn...")

    code.interact(local=dict(globals(), **locals()))


sample = partial(sample_topics, retrieved, questions, qrels, passages)
code.interact(local=dict(globals(), **locals()))

