"""
Question generation using OpenAI models.

This script reads text chunk file in JSONL format, submit question generation
requests in batch to OpenAI API, and save the results in a file once the
OpenAI inference process is completed. 

Usage:
*Step 1: Submit a batch request (if input_file is given)*
    python openai-inference-qgen-textbook.py -input_file <path_to_textbook_chunks.jsonl> --model_name o4-mini-2025-04-16
*Step 2: Retrieve the results (if batch_id is given)*
    python openai-inference-qgen.py --cmd retrieve --batch_id <batch_id>

Question Generation:

Read a chunk from the input file and generate a question. Instruction templates
can be found in the `templates` directory. 

Retrieval:

Once the request is submitted, you can check the status of the batch request
using the batch ID. If the request is completed, the results can be
downloaded and saved to a file.
"""

import code

import os
import random
import json
import argparse

from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv


def load_datasets(input_file):
    """Load dataset from a JSONL file."""
    data = {}
    with open(input_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            data[entry['chunk_id']] = entry
    return data


def create_and_submit_batch_request(
        data,
        model,
        template_type='learning'
):
    # load templates
    fp = 'templates/tmpl_general_learning_qgen.txt'  # for now, one option
    template = open(os.path.join(os.path.dirname(__file__), fp), 'r').read()

    message = {}
    for k, ex in data.items():
        request = [{"role": "system", "content": template}]
        request.append({
            "role": "user",
            "content": f"Context:\n{ex['text']}\nOutput:\n"
        })
        message[ex['chunk_id']] = request

    # save the batch request file in JSONL format.
    batch_fp = "data/openai/batch_request.jsonl"
    with open(batch_fp, 'w') as f:
        for k, row in message.items():
            entry = {
                "custom_id": k,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": row,
                },
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Batch request file saved to {batch_fp}")

    # Uploading batch input file
    print("Submitting OpenAI batch request...")
    batch_file = openai.files.create(
        file=open(batch_fp, "rb"), purpose="batch"
    )
    # Creating the batch request
    batch_resp = openai.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "Question generation for textbook chunks"},
    )
    print(
        f"batch chat completion requested: {batch_resp.id} (cnt. {len(message)})")
    # Save the batch id in a file for future reference
    batch_id_fp = "data/openai/batch_ids.txt"
    with open(batch_id_fp, 'a') as f:
        f.write(batch_resp.id + "\n")
    print(f"Batch ID saved to {batch_id_fp}")


def retrieve_results(batch_id):
    # Check the status and retrieve response if ready
    resp = openai.batches.retrieve(batch_id)

    if resp.status == "completed":
        print("Batch request completed. Fetching results...")

        # Error handling
        if resp.error_file_id is not None:
            print(openai.files.content(resp.error_file_id).read())
            print("Error in batch request. Exiting.")
            exit(1)
        # Download the results
        if resp.output_file_id:
            output_fp = f"data/openai/{args.batch_id}_results.jsonl"
            results = openai.files.content(resp.output_file_id)
            with open(output_fp, "wb") as f:
                f.write(results.read())
            print(f"Results saved to {output_fp}")

    else:
        print(f"Batch job status: {resp.status}")
        exit(0)


def extract_store_questions(input_file, results_file):
    chunks = load_datasets(input_file)
    questions = {}

    # Load the results
    with open(args.results_file, 'r') as f:
        results = [json.loads(line) for line in f.readlines()]
    for res in results:
        chunk_id = res['custom_id']
        # Assuming the last line, prefixed with "Question: ", is the generated
        # question (Accept the line without the prefix as well)
        resp = res['response']['body']['choices'][0]['message']['content'].strip()
        q_str = resp.split('\n')[-1].strip().replace("Questions: ", "")
        question = q_str[len("Question: "):].strip()
        questions[chunk_id] = {
            "question": question,
            "text": chunks[chunk_id]['title'] + ' ' + chunks[chunk_id]['text']
        }

    print(f"Extracted {len(questions)} questions from results.")

    # Save
    with open(f"data/openai/questions.jsonl", 'w') as f:
        for k, v in questions.items():
            entry = {
                "chunk_id": k,
                "question": v['question'],
                "text": v['text']
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Questions saved to data/openai/questions.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", type=str, default="submit",
                        choices=["submit", "retrieve", "extract"],
                        help=("Command to run: submit a batch request, "
                              "retrieve results, or extract questions from results"))
    parser.add_argument("--input_file", type=str,
                        help="File path to the input JSONL file with text chunks")
    parser.add_argument("--model_name", type=str,
                        default="o4-mini-2025-04-16",
                        help="Model name to use for inference")
    parser.add_argument("--batch_id", type=str,
                        help="batch id for OpenAI reponse retrieval")
    parser.add_argument("--results_file", type=str,
                        help="File path to the results for evaluation")

    args = parser.parse_args()

    # Load environment variables, and initialize OpenAI client
    load_dotenv(override=True)
    openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if args.cmd == "submit":
        assert args.input_file, "Input file is required for submission."
        chunks = load_datasets(args.input_file)
        create_and_submit_batch_request(chunks,
                                        model=args.model_name,
                                        template_type='learning')
    elif args.cmd == "retrieve":
        assert args.batch_id, "Batch ID is required for retrieval."
        retrieve_results(args.batch_id)
    elif args.cmd == "extract":
        assert args.results_file, "Results file is required for extraction."
        assert args.input_file, "Input file is required for extraction."
        extract_store_questions(args.input_file, args.results_file)
