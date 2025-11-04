import code

import json
import os
import sys
from typing import List, Dict, Any
import argparse

from tqdm import tqdm
import bm25s
import Stemmer


class BM25_Retriever:
    def __init__(
            self, 
            corpus_name: str, 
            output_dir: str,
            input_file: str = '',
            data_dir: str = 'data/raw'
    ):
        self.corpus_name = corpus_name.lower()
        self.corpus_path = f"{data_dir}/corpora/passage_level/" + \
            f"{self.corpus_name}.jsonl"
        self.questions_path = f"{data_dir}/human/retrieval_tasks/" + \
            f"{self.corpus_name}/{self.corpus_name}_questions.jsonl"
        self.qrels_path = f"{data_dir}/human/retrieval_tasks/" + \
            f"{self.corpus_name}/qrels/dev.tsv"
        self.output_dir = output_dir
        self.input_file = input_file
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
        # - if input_file is provided, use rewritten_queries
        # - if not, use last turns from questions file
        questions = {}
        if self.input_file:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    questions[entry['_id']] = entry['rewritten_queries']
        else:
            with open(self.questions_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    questions[entry['_id']] = entry['text'].split("\n")[-1][8:]

        # retrieve (single questions)
        results = []
        # check if questions is a dict of lists
        if isinstance(list(questions.values())[0], list):
            for qid, qlist in tqdm(questions.items(), desc=f"Retrieving for {self.corpus_name}"):
                results_, scores = self.retriever.retrieve(
                    bm25s.tokenize(qlist, stemmer=self.stemmer), k=top_k
                )
                # consolidate results for multiple questions
                consolidated_contexts = {}
                for (rids, scrs) in zip(results_, scores):
                    for rid, scr in zip(rids, scrs):
                        if rid not in consolidated_contexts:
                            consolidated_contexts[rid] = scr
                        else:
                            consolidated_contexts[rid] += scr
                # sort by score
                sorted_contexts = sorted(
                    consolidated_contexts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:top_k]
                entry = {
                    "conversation_id": qid.split('<::>')[0],
                    "task_id": qid,
                    "turn": qid.split('<::>')[1],
                    "contexts": [
                        {
                            "document_id": self.iid2pid[rid],
                            "score": float(scr)
                        } for rid, scr in sorted_contexts
                    ],
                    "Collection": f"mt-rag-{self.corpus_name}"
                }
                results.append(entry)
        else:
            results_, scores = self.retriever.retrieve(
                bm25s.tokenize(list(questions.values()), stemmer=self.stemmer), k=top_k
            )

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
                    "Collection": f"mt-rag-{self.corpus_name}"
                }
                results.append(entry)

        return results

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run BM25 Retriever on multiple corpora")
    argparser.add_argument('--input_file', type=str, default=None,
                            help='Path to (rewritten) questions file')
    argparser.add_argument('--corpus_name', type=str, default=None,
                           required='--input_file' in sys.argv,
                           choices=['ClapNQ', 'Cloud', 'FiQA', 'Govt'],
                           help='Name of the corpus to use (if input_file is provided)')
    argparser.add_argument('--output_dir', type=str, default='./outputs', 
                           help='Output directory to save indexes and retrieval results')
    args = argparser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    final_results = []

    if args.input_file is None:
        corpora = ['ClapNQ', 'Cloud', 'FiQA', 'Govt']
        for corpus_name in corpora:
            bm25_ret = BM25_Retriever(corpus_name, args.output_dir)
            final_results.extend(bm25_ret.retrieve())
    else:
        bm25_ret = BM25_Retriever(args.corpus_name, args.output_dir,
                                  input_file=args.input_file)
        final_results.extend(bm25_ret.retrieve())

    # save final results as jsonl
    output_path = os.path.join(args.output_dir, 'bm25_retrieval_results.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in final_results:
            f.write(json.dumps(entry) + '\n')
    print(f"Saved retrieval results to {output_path}")

