#!/bin/bash

SOURCE_DIR="../../../mt-rag-benchmark"
CORPORA_DIR="$SOURCE_DIR/corpora"
HUMAN_DIR="$SOURCE_DIR/human"

TARGET_DIR="raw"
mkdir -p "$TARGET_DIR"
ln -s "$CORPORA_DIR" "$TARGET_DIR/corpora"
ln -s "$HUMAN_DIR" "$TARGET_DIR/human"
