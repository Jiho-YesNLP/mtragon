import code

import json
import os
from typing import List, Dict, Any
import argparse

from tqdm import tqdm
import bm25s
import Stemmer


class BM25_Retriever:
    def __init__(self, corpus_name: str, output_dir: str):
        self.corpus_path = f"data/{corpus_name.lower()}.jsonl"
        self.questions_path = f"data/{corpus_name.lower()}-questions.jsonl"
        self.qrels_path = f"data/{corpus_name.lower()}-qrels.tsv"
        self.output_dir = output_dir
        self.passages = self.load_corpus()
        # int id mapping
        self.iid2pid = {i: pid for i, pid in enumerate(self.passages.keys())}
        self.stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(
            [self.passages[self.iid2pid[i]]['text'] for i in sorted(self.iid2pid.keys())], 
            stemmer=self.stemmer
        )
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)


    def load_corpus(self) -> Dict[str, Any]:
        passages = {}
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                passages[entry['id']] = entry
        print(f"Loaded {len(passages):,} passages from {self.corpus_path}")
        return passages


    def retrieve(self, top_k: int = 10) -> List[Dict[str, Any]]:
        # read questions
        questions = []
        with open(self.questions_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                questions.append(entry)

        # last turns
        questions = { q['_id']: q['text'].split("\n")[-1][8:] for q in questions }
        results_, scores = self.retriever.retrieve(
            bm25s.tokenize(list(questions.values()), stemmer=self.stemmer), k=top_k
        )

        results = []
        for (qid, (rids, scrs)) in zip(questions.keys(), zip(results_, scores)):
            entry = {
                "conversation_id": qid.split('<::>')[0],
                "task_id": qid,
                "turn": qid.split('<::>')[1],
                "contexts": [
                    {
                        "document_id": self.iid2pid[rid],
                        "score": float(scr)
                    } for rid, scr in zip(rids, scrs)
                ],
                "Collection": f"mt-rag-{corpus_name.lower()}"
            }
            results.append(entry)

        return results

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run BM25 Retriever on multiple corpora")
    argparser.add_argument('--output-dir', type=str, default='./outputs', 
                           help='Output directory to save indexes and retrieval results')
    args = argparser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    corpora = ['ClapNQ', 'Cloud', 'FiQA', 'Govt']

    final_results = []
    for corpus_name in corpora:
        bm25_ret = BM25_Retriever(corpus_name, args.output_dir)
        final_results.extend(bm25_ret.retrieve())

    # save final results as jsonl
    output_path = os.path.join(args.output_dir, 'bm25_retrieval_results.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in final_results:
            f.write(json.dumps(entry) + '\n')
    print(f"Saved retrieval results to {output_path}")

