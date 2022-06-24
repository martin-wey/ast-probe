import logging
import argparse
import random
import os

import numpy as np
import networkx as nx
import torch
from transformers import AutoTokenizer
from tree_sitter import Parser
from datasets import load_dataset

from data import download_codesearchnet_dataset, PY_LANGUAGE, JS_LANGUAGE, GO_LANGUAGE, \
    PHP_LANGUAGE, JAVA_LANGUAGE, RUBY_LANGUAGE
from data.code2ast import code2ast, get_tokens_ast, has_error
from data.utils import match_tokenized_to_untokenized_roberta

tokenizer_roberta = AutoTokenizer.from_pretrained('roberta-base')
tokenizer_t5 = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
tokenizer_codebert = AutoTokenizer.from_pretrained('microsoft/codebert-base')
tokenizer_graphcodebert = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')
tokenizer_codeberta = AutoTokenizer.from_pretrained('huggingface/CodeBERTa-small-v1')

tokenizers = [tokenizer_roberta, tokenizer_t5, tokenizer_codebert, tokenizer_graphcodebert, tokenizer_codeberta]


def filter_samples(code, max_length, lang, parser):
    try:
        G, code_pre = code2ast(code=code, parser=parser, lang=lang)
        assert nx.is_tree(nx.Graph(G))
        assert nx.is_connected(nx.Graph(G))
    except:
        return False
    if has_error(G):
        return False

    for tokenizer in tokenizers:
        t, _ = match_tokenized_to_untokenized_roberta(untokenized_sent=code_pre, tokenizer=tokenizer)
        if len(t) + 2 > max_length:
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for generating the dataset for probing')
    parser.add_argument('--dataset_dir', default='./dataset', help='Path to save the dataset')
    parser.add_argument('--lang', help='Language.', choices=['javascript', 'python', 'go', 'php', 'java', 'ruby'],
                        default='python')
    parser.add_argument('--max_code_length', help='Maximum code length.', default=512)
    parser.add_argument('--download', help='If download the csn', action='store_true')
    parser.add_argument('--seed', help='seed.', type=int, default=123)
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    # seed everything
    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # download dataset
    if args.download:
        download_codesearchnet_dataset(dataset_dir=args.dataset_dir)
    dataset_path = os.path.join(args.dataset_dir, args.lang, 'dataset.jsonl')
    logger.info('Loading dataset.')
    dataset = load_dataset('json', data_files=dataset_path, split='train')

    # select the parser
    parser_lang = Parser()
    if args.lang == 'python':
        parser_lang.set_language(PY_LANGUAGE)
    elif args.lang == 'javascript':
        parser_lang.set_language(JS_LANGUAGE)
    elif args.lang == 'go':
        parser_lang.set_language(GO_LANGUAGE)
    elif args.lang == 'php':
        parser_lang.set_language(PHP_LANGUAGE)
    elif args.lang == 'java':
        parser_lang.set_language(JAVA_LANGUAGE)
    elif args.lang == 'ruby':
        parser_lang.set_language(RUBY_LANGUAGE)

    # filter dataset
    logger.info('Filtering dataset.')
    dataset = dataset.filter(
        lambda e: filter_samples(e['original_string'], args.max_code_length, args.lang, parser_lang), num_proc=6)

    logger.info('Shuffling dataset.')
    dataset = dataset.shuffle(args.seed)

    logger.info('Splitting dataset.')
    if args.lang == 'ruby':
        train_dataset = dataset.select(range(0, 10000))
        test_dataset = dataset.select(range(10000, 12000))
        val_dataset = dataset.select(range(12000, 13000))
    else:
        train_dataset = dataset.select(range(0, 20000))
        test_dataset = dataset.select(range(20000, 24000))
        val_dataset = dataset.select(range(24000, 26000))

    logger.info('Storing dataset.')
    train_dataset.to_json(os.path.join(args.dataset_dir, args.lang, 'train.jsonl'))
    test_dataset.to_json(os.path.join(args.dataset_dir, args.lang, 'test.jsonl'))
    val_dataset.to_json(os.path.join(args.dataset_dir, args.lang, 'valid.jsonl'))
