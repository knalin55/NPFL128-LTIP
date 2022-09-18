#!/usr/bin/env python3

import numpy as np
from random import shuffle
from transformers import AutoTokenizer
import random
from corpus_reader.benchmark_reader import Benchmark, select_files
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset
import random
from datasets import concatenate_datasets

import argparse
import datetime
import os
import re
from typing import Dict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
	parser.add_argument("--finetune_type", default=None, type=str, help="Pretraining method")
	parser.add_argument("--pretrain_type", default="mt5_wmt", type=str, help="Pretraining method")
	parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight Decay.")
	parser.add_argument("--epochs", default=25, type=int, help="Number of epochs.")
	parser.add_argument("--steps", default=200000, type=int, help="Number of epochs.")
	parser.add_argument("--lr", default=2e-5, type=float, help="Learning Rate.")
	parser.add_argument("--model", default="t5-small", type=str, help="Pretrained-model")
	parser.add_argument("--seed", default=42, type=int, help="Random seed.")
	parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
	parser.add_argument("--lang", default="ru", type=str, help="Language (Relevant for monolingual finetuning and choosing dev set)")
	parser.add_argument("--ckpt", default=None, type=str, help="Resume from checkpoint")
	return parser

def camel_case_split(str):
	return [re.findall(r"[a-z]*", str)[0]] + re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str) 

def process_triples(fil_dir, lang="en"):
          
    b = Benchmark()
    files = select_files(fil_dir)
    b.fill_benchmark(files)
      
    entry_list = [entry for entry in b.entries]
    triples_list    = []
    text_list       = []

    sub = "<S>"
    obj = "<O>"
    pre = "<P>"

    # Parallel data for sentences and entities
    en_sent = []
    ru_sent = []
    en_enti = []
    ru_enti = []
        
    random.shuffle(entry_list)

    for entry in entry_list:
        
        # entry.list_triples : [<subj, pred, obj>]
        # entry.lexs[i].lex : Sentence (for ru dataset, Sentence belongs to ru if i is odd else en)
        # entry.links : [<eng_entity, relation, rus_entity>]


        if lang == "ru":
            for link in entry.links:
                en_enti.append(link.s)
                ru_enti.append(link.o)
            en_sent = en_sent + [entry.lexs[i].lex for i in range(len(entry.lexs)) if i % 2 == 0]
            ru_sent = ru_sent + [entry.lexs[i].lex for i in range(len(entry.lexs)) if i % 2 == 1]
        
        lang_ = 0 # 0 for English | 1 for Russian
        
        for text in entry.lexs:
            
            label = ""
            if (lang == "ru" and lang_ % 2 == 1) or lang == "en":
                for triple in entry.list_triples():
                    
                    # e.g. Aarhus_Airport | elevationAboveTheSeaLevel | 25.0
                    # split using "|" -> get subj, pred, obj
                    # remove "_" for subj and obj, and remove camelCase for pred
                    # append processed triple and corresponding text to input and label

                    inputs = text.lex
                    label = label + " " + sub  + " " +  " ".join(triple.split("|")[0].strip().split("_")) \
                         + " " +  pre  + " " +   " ".join([word.lower() for word in camel_case_split(triple.split("|")[1].strip())])  + " " +  \
                            obj  + " " +  " ".join(triple.split("|")[2].strip().split("_"))
                    
                triples_list.append(label.strip())
                text_list.append(inputs)

            lang_ += 1

    return {"input":triples_list, "label":text_list} if lang == "en" else \
        {"input":triples_list, "label":text_list, "en_sent": en_sent, "ru_sent": ru_sent, "en_entity": en_enti, "ru_entity": ru_enti}


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    def preprocess_function(examples):

        # add prefix for each task as done in T5 paper
        input = [examples["prefix"][i] + " " + example for i, example in enumerate(examples["input"])]
        label = examples["input"] if len(examples.features) == 1 else examples["label"] 
        model_inputs = tokenizer(input, max_length=512, padding=True, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(label, max_length=512, padding=True, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    prefix = dict(en= "Generate in English", ru= "Generate in Russian")

    if args.finetune_type == "monolingual":
        
        # Monolingual: train separate models for both languages
        # Preparing dataset for args.lang language

        data = process_triples("./data/webnlg_dataset/release_v3.0/{}/train".format(args.lang), args.lang)
        dataset = Dataset.from_dict({"input":data["input"], "label":data["label"], "prefix": [prefix[args.lang]]*len(data["input"])})


    elif args.finetune_type == "bilingual":

        # Bilingual: train single models for both languages
        # Preparing dataset for both languages and concatenating them


        data_en = process_triples("./data/webnlg_dataset/release_v3.0/en/train", "en")
        data_ru = process_triples("./data/webnlg_dataset/release_v3.0/ru/train", "ru")
        data_en["prefix"] = [prefix["en"]]*len(data_en["input"])
        dataset_en = Dataset.from_dict(data_en)
        dataset_ru = Dataset.from_dict({"input":data_ru["input"], "label":data_ru["label"], "prefix": [prefix["ru"]]*len(data_ru["input"])})
        dataset = concatenate_datasets([dataset_en, dataset_ru])

    elif args.finetune_type == "bilingual+wpc":

        # Bilingual: train single models for both languages (along with using WebNLG parallel data)
        
        # Preparing dataset for Data-to-Text generation (en+ru), sentence translation (en -> ru + ru -> en)\
            #   and entity translation (en -> ru + ru -> en) and concatenating them


        data_en = process_triples("./data/webnlg_dataset/release_v3.0/en/train", "en")
        data_ru = process_triples("./data/webnlg_dataset/release_v3.0/ru/train", "ru")
        data_en["prefix"] = [prefix["en"]]*len(data_en["input"])
        dataset_en = Dataset.from_dict(data_en)
        dataset_ru = Dataset.from_dict({"input":data_ru["input"], "label":data_ru["label"], "prefix": [prefix["ru"]]*len(data_ru["input"])})
        
        prefix_en = "Translate to English"
        prefix_ru = "Translate to Russian"

        wpc_sent_en_ru = Dataset.from_dict({"input":data_ru["en_sent"], "label":data_ru["ru_sent"], "prefix":[prefix_ru]*len(data_ru["ru_sent"])})
        wpc_sent_ru_en = Dataset.from_dict({"input":data_ru["ru_sent"], "label":data_ru["en_sent"], "prefix":[prefix_en]*len(data_ru["ru_sent"])})
        
        wpc_enti_en_ru = Dataset.from_dict({"input":data_ru["en_entity"], "label":data_ru["ru_entity"], "prefix":[prefix_ru]*len(data_ru["ru_entity"])})
        wpc_enti_ru_en = Dataset.from_dict({"input":data_ru["ru_entity"], "label":data_ru["en_entity"], "prefix":[prefix_en]*len(data_ru["ru_entity"])})
        
        dataset = concatenate_datasets([dataset_en, dataset_ru, wpc_sent_en_ru, wpc_sent_ru_en, wpc_enti_en_ru, wpc_enti_ru_en])
        


    dataset = dataset.map(preprocess_function, batched=True)
    dataset = dataset.shuffle()
    train_dataset = dataset.remove_columns(["input", "label", "prefix"])


    #Preparing validation data

    val_data = process_triples("./data/webnlg_dataset/release_v3.0/{}/dev".format(args.lang), args.lang)
    val_dataset = Dataset.from_dict({"input":val_data["input"], "label":val_data["label"], "prefix": [prefix[args.lang]]*len(val_data["input"])})
    
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.shuffle()
    val_dataset = val_dataset.remove_columns(["input", "label", "prefix"])


    # Loading model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding = True, return_tensors="pt")

    training_args = Seq2SeqTrainingArguments(
        output_dir= os.path.join("check_points/{}/{}_{}/".format(args.pretrain_type, args.finetune_type, args.lang), args.logdir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy = "steps",
        save_strategy= "steps",
        eval_steps= 10000, 
        save_total_limit=2)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator)


    if args.ckpt is not None:
        trainer.train(resume_from_checkpoint=args.ckpt)
    else:
        trainer.train()


if __name__ == "__main__":
    args = get_parser().parse_args([] if "__file__" not in globals() else None)
    main(args)
