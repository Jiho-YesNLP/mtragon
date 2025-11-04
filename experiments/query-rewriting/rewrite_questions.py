"""
Utilize an LLM to rewrite incomplete and ambiguous user questions into clear
and complete search queries and candidate answers to the questions.

The question rewriting process involves three main steps and a final step to
generate answer to the question.

Step 1: Clarification; correct any spelling errors and identify missing topics
Step 2: Expansion; add relevant context and details to the question
Step 3: Reformulation; rephrase the question into a clear and concise query
Step 4: Answer Generation; generate a candidate answer to the rewritten question
"""

import code
import os
import argparse
import json
import random

import openai
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)


def build_system_prompt() -> str:
    return (
        "You are an expert question rewriting assistant that helps a "
        "retrieval system. Your task is to rewrite a user's question within a "
        "conversation, which may be incomplete or ambiguous, into a clear and "
        "complete search query. \n\n"
        "The rewriting process involves three main steps:\n"
        "1. Clarification: Correct any spelling errors and identify missing "
        "topics or context needed to fully understand the question.\n"
        "2. Expansion: Add relevant context and details to the question to "
        "make it more specific and informative.\n"
        "3. Reformulation: Rephrase the question into a clear and concise "
        "search query that accurately reflects the user's intent.\n"
        "After rewriting the question, also generate a candidate answer to the "
        "rewritten question based on your knowledge.\n\n"
        "Primary considerations include:\n"
        "- Ensure the rewritten question is self-contained and understandable "
        "without additional context.\n"
        "- Maintain the original intent of the user's question while enhancing "
        "its clarity and completeness.\n"
        "- The list of expansion terms should be relevant and directly related "
        "to the topic of the question.\n"
        "- Resolve coreferences to make the question unambiguous.\n"
        "- Avoid introducing new topics not present in the original question.\n"
    )

def build_user_prompt(pid, conv_id, turn_id, contexts, cq_no: int = 1) -> str:
    schema_step1 = { "clarified_questions": [ "string" ] }
    schema_step2 = { "expansion_terms": [ "string" ] }
    question = contexts[conv_id][turn_id]['question']
    if pid == 1:  # Step 1: Clarification
        if turn_id == 1 or (turn_id - 1) not in contexts[conv_id]:
            context = ""
        else:
            context = "Previous Question: {}\n Answer: {}".format(
                contexts[conv_id][turn_id - 1]['question'],
                contexts[conv_id][turn_id - 1].get('answer', 'N/A')
            )
        prompt = (  
            "Step 1: Clarification\n\n"
            "Given the context of the conversation and the original question, "
            "which may be incomplete or ambiguous, correct any spelling errors and "
            "identify any missing topics or context needed to fully understand the "
            "question. Provide three clarified version of the question."
            "Output strictly as minified JSON matching this schema: \n"
            + json.dumps(schema_step1, separators=(',', ':')) + "\n\n"
            f"{context}\n"
            f"Original Question: \"{question}\"\n"
            "Provide three clarified versions of the question."          
        )
    elif pid == 2:  # Step 2: Expansion
        prompt = (
            "Step 2: Expansion\n\n"
            "Given the a clarified question from Step 1, provide a list of "
            "related concepts and terms that can be used to expand the "
            "question for a more comprehensive search. Provide up to five "
            "expansion terms in the order of relevance."
            "Output strictly as minified JSON matching this schema: \n"
            + json.dumps(schema_step2, separators=(',', ':')) + "\n\n"
            f"Question: {question}\n"
            "Provide up to five expansion terms."
        )
    elif pid == 3:  # Step 3: Reformulation
        cq = contexts[conv_id][turn_id]['clarified_questions'][cq_no]
        eterms = random.sample(
            contexts[conv_id][turn_id]['expansion_terms'],
            min(5, len(contexts[conv_id][turn_id]['expansion_terms']))
        )
        prompt = (
            "Step 3: Reformulation\n\n"
            "Using the clarified question from Step 1 and the expansion "
            "terms from Step 2, rephrase the question into a clear and concise "
            "search query that accurately reflects the user's intent. "
            "Output strictly as a string representing the rewritten question.\n\n"
            f"Question: {question}\n"
            f"Clarified Questions: {cq}\n"
            f"Expansion Terms: {', '.join(eterms)}\n"
            "Provide the reformulated query."
        )
    elif pid == 4:  # Step 4: Answer Generation
        cq = '\n  - '.join(contexts[conv_id][turn_id]['clarified_questions'])
        eterms = contexts[conv_id][turn_id]['expansion_terms']
        prompt = (
            "Step 4: Answer Generation\n\n"
            "Based on the rewritten questions from Step 3, generate a "
            "concise and informative candidate answer to the original question. "
            "Output strictly as a string representing the answer.\n\n"
            f"Original Question: {question}\n"
            f"Rewritten Questions: \n  - {cq}\n"
            "Expansion Terms: {eterms}\n"
            "Provide the candidate answer."
        )
    else:
        raise NotImplementedError("Only step 1 is implemented.")
    return prompt
    

def rewrite_question( llm, model, conv_id, turn_id, contexts) -> dict:
    system_prompt = build_system_prompt()
    # Step 1: Clarification
    print(f"{conv_id}::{turn_id} - Step 1: Clarification\r", end='', flush=True)
    user_prompt = build_user_prompt(1, conv_id, turn_id, contexts)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = llm.chat.completions.create(
        model=model,
        messages=messages,
    )
    clarified_questions = json.loads(
        response.choices[0].message.content
    )['clarified_questions']
    contexts[conv_id][turn_id]['clarified_questions'] = clarified_questions

    # Step 2: Expansion
    print(f"{conv_id}::{turn_id} - Step 2: Expansion    \r", end='', flush=True)
    expansion_terms = []
    for cq in clarified_questions:
        user_prompt = build_user_prompt(2, conv_id, turn_id, contexts)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = llm.chat.completions.create(
            model=model,
            messages=messages,
        )
        terms = json.loads(response.choices[0].message.content)['expansion_terms']
        expansion_terms.extend([t.strip().lower() for t in terms])
    contexts[conv_id][turn_id]['expansion_terms'] = list(set(expansion_terms))

    # Step 3: Reformulation
    print(f"{conv_id}::{turn_id} - Step 3: Reformulation\r", end='', flush=True)
    for cq_no in range(len(clarified_questions)):
        user_prompt = build_user_prompt(3, conv_id, turn_id, contexts, cq_no)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = llm.chat.completions.create(
            model=model,
            messages=messages,
        )
        final_query = response.choices[0].message.content.strip()
        contexts[conv_id][turn_id].setdefault('queries', []).append(final_query)

    # Step 4: Answer Generation
    print(f"{conv_id}::{turn_id} - Step 4: Answer Generation\r", end='', flush=True)
    user_prompt = build_user_prompt(4, conv_id, turn_id, contexts)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = llm.chat.completions.create(
        model=model,
        messages=messages,
    )
    answer = response.choices[0].message.content.strip()
    contexts[conv_id][turn_id]['answer'] = answer
    print(f"{conv_id}::{turn_id} - Done." + " " * 20)
    return response
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_file", type=str, required=True,
        help="Path to the JSONL file containing original questions.",
    )
    parser.add_argument("--output_file", type=str, required=True,
        help="Path to the file to save rewritten questions and answers.",
    )
    parser.add_argument("--model_name", type=str, default=None,
        help="Name of the OpenAI model to use for rewriting.",
    )
    args = parser.parse_args()

    llm = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                        base_url=os.getenv("OPENAI_API_BASE_URL"))

    model = 'gpt-5'
    try:
        llm.models.list()
    except openai.AuthenticationError:
        print("API key is invalid.")
    else:
        print("API key is valid.")
        if args.model_name:
            model = args.model_name
        else:
            # print list of available models
            models = llm.models.list()
            for i, model in enumerate(models.data):
                print(f"{i+1}. {model.id}")
            # select a model
            model_id = input("Select a model by number: ")
            model = models.data[int(model_id) - 1].id
            print(f"Selected model: {model}")

    # Read questions from file
    contexts = {}  # 'conv_id<::>turn_id'
    with open(args.questions_file, "r") as f:
        for line in f:
            item = json.loads(line)
            conv_id, turn_id = item['_id'].split('<::>')
            if conv_id not in contexts:
                contexts[conv_id] = {}
            question = item['text'].split('\n')[-1][len('|user|: '):].strip()
            contexts[conv_id][int(turn_id)] = { 'question': question }
    print(f"Loaded {len(contexts)} conversations.")
    
    for conv_id in contexts:
        # sort turn ids
        turns = sorted(contexts[conv_id].keys())
        turns = [int(t) for t in turns]
        for turn_id in turns:
            question = contexts[conv_id][turn_id]['question']
            rewrite_question(llm, model, conv_id, turn_id, contexts)
            with open(args.output_file, "a") as out_f:
                entry = {
                    "_id": f"{conv_id}<::>{turn_id}",
                    "question": contexts[conv_id][turn_id]['question'],
                    "clarified_questions": contexts[conv_id][turn_id].get('clarified_questions', []),
                    "expansion_terms": contexts[conv_id][turn_id].get('expansion_terms', []),
                    "rewritten_queries": contexts[conv_id][turn_id].get('queries', []),
                    "answer": contexts[conv_id][turn_id].get('answer', '')
                }
                out_f.write(json.dumps(entry) + "\n")

    code.interact(local=dict(globals(), **locals()))
