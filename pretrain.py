#!/usr/bin/env python3

import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset
import random
from datasets import load_dataset, concatenate_datasets

import argparse
import datetime
import os
import re
from typing import Dict

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
parser.add_argument("--pretrain_type", default=None, type=str, help="Pretraining method")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight Decay.")
parser.add_argument("--steps", default=200000, type=int, help="Number of steps.")
parser.add_argument("--lr", default=2e-5, type=float, help="Learning Rate.")
parser.add_argument("--model", default="t5-small", type=str, help="Seq to Seq Model")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--ckpt_path", default=None, type=str, help="Checkpoint Path to resume from")


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    
    


    if args.pretrain_type == "bt5":

        # BT5: Training T5 on en and ru Wiki dump. 
        # load parquet files for both languages

        dataset_en = load_dataset("parquet", data_files="data/en.parquet")
        dataset_ru = load_dataset("parquet", data_files="data/ru.parquet")
        dataset_en = dataset_en["train"].remove_columns([col for col in dataset_en["train"].column_names if col != "input"])
        dataset_ru = dataset_ru["train"].remove_columns([col for col in dataset_ru["train"].column_names if col != "input"])
        assert dataset_en.features.type == dataset_ru.features.type
        
        dataset = concatenate_datasets([dataset_en, dataset_ru])
        dataset = dataset.add_column("label", dataset["input"])

        def get_training_corpus():
            for start_idx in range(0, len(dataset), 1000):
                samples = dataset[start_idx : start_idx + 1000]
                yield samples["input"]


    elif args.pretrain_type == "bt5_wmt" or args.pretrain_type == "mt5_wmt":

        # BT5_WMT/MT5_WMT: Training T5/MT5 further on WMT dataset. 
        # load and process tsv file for WMT News dataset

        dataset = open("./data/news-commentary-v14.en-ru.tsv").read().split("\n")
        en = []
        ru = []
        for line in dataset:
            en.append(line.split("\t")[0])
            ru.append(line.split("\t")[1])

        parallel_data = {"input": en, "label": ru}    

        dataset = Dataset.from_dict(parallel_data)
        
        def get_training_corpus():
            data = dataset["input"] + dataset["label"]
            random.shuffle(data)
            for start_idx in range(0, len(data), 1000):
                samples = data[start_idx : start_idx + 1000]
                yield samples


    elif args.pretrain_type == "bt5+wmt":

        # BT5+WMT: Training T5 on Wiki dump + WMT dataset. 
        # concatenate datasets from all three sources

        dataset_en = load_dataset("parquet", data_files="data/en.parquet")
        dataset_ru = load_dataset("parquet", data_files="data/ru.parquet")
        dataset_en = dataset_en["train"].remove_columns([col for col in dataset_en["train"].column_names if col != "input"])
        dataset_ru = dataset_ru["train"].remove_columns([col for col in dataset_ru["train"].column_names if col != "input"])
        assert dataset_en.features.type == dataset_ru.features.type
        
        wiki_dataset = concatenate_datasets([dataset_en, dataset_ru])

        wiki_dataset = wiki_dataset.add_column("label", wiki_dataset["input"])

        dataset = open("./data/news-commentary-v14.en-ru.tsv").read().split("\n")
        en = []
        ru = []
        for line in dataset:
            en.append(line.split("\t")[0])
            ru.append(line.split("\t")[1])

        parallel_data = {"input": en, "label": ru}    

        parallel_dataset = Dataset.from_dict(parallel_data)

        dataset = concatenate_datasets([wiki_dataset, parallel_dataset])
        
        def get_training_corpus():
            data = en + ru + wiki_dataset["input"]
            random.shuffle(data)
            for start_idx in range(0, len(data), 1000):
                samples = data[start_idx : start_idx + 1000]
                yield samples



    #Load and train tokenizer
    if args.pretrain_type != "mt5_wmt":
        tokenizer_training_corpus = get_training_corpus()
        old_tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer = old_tokenizer.train_new_from_iterator(tokenizer_training_corpus, 32128)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Preprocessing function 
    def preprocess_function(examples):

        input = examples["input"]
        label = examples["input"] if len(examples.features) == 1 else examples["label"] 
        model_inputs = tokenizer(input, max_length=512, padding=True, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(label, max_length=512, padding=True, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = dataset.map(preprocess_function, batched=True)
    dataset = dataset.shuffle()
    dataset = dataset.remove_columns("input")
    if "label" in dataset.features: dataset = dataset.remove_columns("label")
    dataset = dataset.train_test_split(test_size=0.02)

    
    # Load Model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # Data Collator 
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding = True, return_tensors="pt")


    training_args = Seq2SeqTrainingArguments(
        output_dir= os.path.join(args.logdir, "check_points/{}/".format(args.pretrain_type)),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.steps,
        weight_decay=args.weight_decay,
        evaluation_strategy = "steps",
        save_strategy= "steps",
        eval_steps= 10000, 
        save_total_limit=2)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator)

    if args.ckpt_path is not None:
        trainer.train(resume_from_checkpoint=args.ckpt_path)
    else:
        trainer.train()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)