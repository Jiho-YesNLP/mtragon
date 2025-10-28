"""
This script reads the raw corpora datasets in jsonl files and creat the
combined corpora.jsonl file, which will be used in indexing process.

Each entry in the input file contains the following fields: 
    _id, id, text, title, url

- _id and id are identical.
- some files don't have title or url values. 
"""

import code
import os
import json



if __name__ == "__main__":
    # TODO. add options to customize chunking and preprocessing

    corpus = {}

    # read all jsonl files
    raw_data_dir = "data/raw/corpora/passage_level/"
    for file in os.listdir(raw_data_dir):
        if file.endswith(".jsonl"):
            corpus_name = file.split(".")[0]
            with open(os.path.join(raw_data_dir, file), "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    _id = entry["_id"]
                    text = entry["text"].strip()
                    title = entry.get("title", "").strip()
                    url = entry.get("url", "").strip()
                    
                    # Ensure no duplicate ids
                    if _id in corpus:
                        raise ValueError(f"Duplicate id found: {_id}")
                    
                    corpus[_id] = {
                        "contents": text,
                        "title": title,
                        "corpus": corpus_name,
                    }
                    

    # save the processed data
    output_path = "data/processed/corpora.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for _id, entry in corpus.items():
            json_line = {
                "_id": _id,
                "contents": entry["contents"],
                "title": entry["title"],
                "corpus": entry["corpus"],
            }
            f.write(json.dumps(json_line) + "\n")
