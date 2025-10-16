"""
Use FAISS for approximate nearest neighbor search with DPR embeddings.
The retriver will use a pre-trained question/context encoder to encode corpora
and queries

How to run:
- encoder-name: facebook/dpr-ctx_encoder-single-nq-base
"""

import code
import argparse
import os
import json
from collections import defaultdict
from tqdm import tqdm

import faiss
import torch
import numpy as np
import pandas as pd

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

class FaissRepresentationWriter():
    def __init__(self, dir_path, dimension=768):
        self.dir_path = dir_path
        self.index_name = 'index'
        self.id_file_name = 'pid'
        self.dim = dimension
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_file = None

    def __enter__(self):
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        self.id_file = open(os.path.join(self.dir_path, self.id_file_name), 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.id_file is not None:
            self.id_file.close()
        faiss.write_index(self.index, os.path.join(self.dir_path, self.index_name))

    def write(self, batch_info, fields=None):
        if self.id_file is not None:
            for id_ in batch_info['id']:
                self.id_file.write(f'{id_}\n')
        self.index.add(np.ascontiguousarray(batch_info['vector']))


class DprDocumentEncoder:
    def __init__(self, model_name, tokenizer_name=None, device='cuda:0'):
        self.device = device
        self.model = DPRContextEncoder.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            tokenizer_name or model_name,
            clean_up_tokenization_spaces=True
        )

    def encode(self, texts, max_length=256, **kwargs):
        inputs = self.tokenizer(
            texts,
            max_length=max_length,
            padding='longest',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        inputs.to(self.device)

        # pooler_output contains the hidden representations for each text
        return self.model(inputs["input_ids"]).pooler_output.detach().cpu().numpy()


class DprQueryEncoder:
    def __init__(self, encoder_dir: str = "", tokenizer_name: str = "",
                 encoded_query_dir: str = "", device: str = 'cpu', **kwargs):
        self.has_model = False
        self.has_encoded_query = False
        if encoded_query_dir:
            df = pd.read_pickle(
                os.path.join(encoded_query_dir, 'embedding.pkl')
            )
            self.embedding = dict(zip(df['text'].tolist(), df['embedding'].tolist()))
            self.has_encoded_query = True


        self.device = device
        if encoder_dir:
            self.model = DPRQuestionEncoder.from_pretrained(encoder_dir)
            self.model.to(self.device)
            self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
                tokenizer_name or encoder_dir,
                clean_up_tokenization_spaces=True
            )
            self.has_model = True
        if (not self.has_model) and (not self.has_encoded_query):
            raise Exception('Neither query encoder model nor encoded queries provided. Please provide at least one')

    def encode(self, query: str):
        if self.has_model:
            input_ids = self.tokenizer(query, return_tensors='pt')
            input_ids.to(self.device)
            embeddings = self.model(input_ids["input_ids"]).pooler_output.detach().cpu().numpy()
            return embeddings.flatten()
        else:
            return super().encode(query)


class DPR_Retriever:
    def __init__(self, encoder: DprQueryEncoder, corpus_name: str, index_dir: str):
        self.encoder = encoder
        self.questions = self.load_questions(corpus_name)
        self.index = faiss.read_index(
            os.path.join(index_dir, corpus_name.lower(), 'index')
        )
        self.dimension = self.index.d
        self.num_docs = self.index.ntotal
        id_f = open(os.path.join(index_dir, corpus_name.lower(), 'pid'), 'r')
        self.pids = [l.rstrip() for l in id_f.readlines()]
        assert len(self.pids) == self.num_docs

    def load_questions(self, corpus_name: str):
        questions_path = f"data/{corpus_name.lower()}-questions.jsonl"
        questions = []
        with open(questions_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                questions.append(entry)
        # last turns
        questions = { q['_id']: q['text'].split("\n")[-1][8:] for q in questions }
        return questions

    def retrieve(self, top_k: int = 10):
        results = []
        for qid, question in tqdm(self.questions.items()):
            q_vector = self.encoder.encode(question).reshape(1, self.dimension)
            scores, indices = self.index.search(q_vector, top_k)
            retrieved = [
                {'document_id': self.pids[idx], 'score': float(score)}
                for score, idx in zip(scores[0], indices[0])
            ]
            results.append({
                'conversation_id': qid.split('<::>')[0],
                'task_id': qid,
                'turn': qid.split('<::>')[1],
                'contexts': retrieved,
                'Collection': f'mt-rag-{corpus_name.lower()}'
            })
        return results


if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser(description="DPR Retriever")
    parser.add_argument('--ctx-encoder', type=str,
                        default='facebook/dpr-ctx_encoder-single-nq-base',
                        help='Encoder name or path for encoding contexts')
    parser.add_argument('--q-encoder', type=str,
                        default='facebook/dpr-question_encoder-single-nq-base',
                        help='Encoder name or path for encoding questions')
    parser.add_argument('--encoder-class', type=str, default='dpr',
                        choices=['dpr', 'sentence-transformers'],
                        help='Encoder class to use')
    parser.add_argument('--dim', type=int, default=768,
                        help='Dimension of the embeddings')
    parser.add_argument('--output-dir', type=str, default='./outputs/dpr',
                        help='Directory to store encoded corpus')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for encoding')
    parser.add_argument('--max-length', type=int, default=256,
                        help='Max length for encoding')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use fp16 for encoding')
    parser.add_argument('--add-sep', action='store_true', default=False,
                        help='Add separator token between sentences')

    args = parser.parse_args()

    # encoder
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    encoder = None

    # read corpora
    corpora = ['ClapNQ', 'Cloud', 'FiQA', 'Govt']
    passages = defaultdict(dict)
    enc_args = {
        'fp16': args.fp16,
        'max_length': args.max_length,
        'add_sep': args.add_sep,
    }
    for corpus_name in corpora:
        with open(f'./data/{corpus_name.lower()}.jsonl', 'r', encoding='utf-8') as f:
            for l in f:
                entry = json.loads(l)
                passages[corpus_name][entry['id']] = entry
        print(f'Loaded {len(passages[corpus_name])} passages for corpus {corpus_name}')
        # if faiss index exists, skip encoding
        if os.path.exists(os.path.join(args.output_dir, corpus_name.lower(), 'index')):
            print(f'Faiss index for corpus {corpus_name} already exists, skipping encoding...')
            continue
        if encoder is None:
            encoder = DprDocumentEncoder(args.ctx_encoder, device=device)

        emb_writer = FaissRepresentationWriter(
            os.path.join(args.output_dir, corpus_name.lower()),
            dimension=args.dim
        )
        # create representations if not exist
        with emb_writer:
            print(f'Encoding corpus {corpus_name}...')
            batch_texts = []
            batch_ids = []
            for pid, passage in tqdm(passages[corpus_name].items()):
                batch_texts.append(passage['text'])
                batch_ids.append(pid)
                if len(batch_texts) == args.batch_size:
                    vectors = encoder.encode(texts=batch_texts, **enc_args)
                    emb_writer.write({'id': batch_ids, 'vector': vectors})
                    batch_texts = []
                    batch_ids = []
            if batch_texts:
                vectors = encoder.encode(batch_texts)
                emb_writer.write({'id': batch_ids, 'vector': vectors})
            print(f'Finished encoding corpus {corpus_name}')


    # query encoder
    encoder = DprQueryEncoder(args.q_encoder)

    # retrieve for evaluation
    final_results = []
    for corpus_name in corpora:
        dpr_ret = DPR_Retriever(encoder, corpus_name, args.output_dir)
        final_results.extend(dpr_ret.retrieve(top_k=10))

    # save final results as jsonl
    output_path = os.path.join(args.output_dir, 'dpr_retrieval_results.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in final_results:
            f.write(json.dumps(entry) + '\n')
    print(f"Saved retrieval results to {output_path}")

    code.interact(local=dict(globals(), **locals()))
    #
