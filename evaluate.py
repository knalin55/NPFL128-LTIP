#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from corpus_reader.benchmark_reader import Benchmark, select_files
from tqdm import tqdm 
from sacrebleu.metrics import BLEU
import argparse
import re
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", default=None, type=str, help="Pretrained model")
    parser.add_argument("--finetune", default=None, type=str, help="Number of lines to keep in dataset")
    parser.add_argument("--ckpt", default=None, type=str, help="Number of lines to keep in dataset")
    return parser

def main(args: argparse.Namespace) -> None:
    
    langs = ["en", "ru"]
    if args.finetune == "monolingual_ru":
        langs = ["ru"]
    elif args.finetune == "monolingual_en":
        langs = ["en"]

    for lang in langs:
    
        fil_dir = "./data/webnlg_dataset/release_v3.0/{}/dev".format(lang)
        prefix = dict(en= "Generate in English ", ru= "Generate in Russian ")

        def camel_case_split(str):
            return [re.findall(r"[a-z]*", str)[0]] + re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str) 

        b = Benchmark()
        files = select_files(fil_dir)
        b.fill_benchmark(files)
            
        entry_list = [entry for entry in b.entries]
        triples_list    = [] # Inputs to model (RDF Triples)
        text_list       = [] # Labels (Text)

        sub = "<S>"
        obj = "<O>"
        pre = "<P>"


        for entry in entry_list:
            
            inputs = ""
            label = [sent.lex for i, sent in enumerate(entry.lexs) if i%2==1] if lang == "ru" else [sent.lex for i, sent in enumerate(entry.lexs)]
            for triple in entry.list_triples():
                
                
                inputs = inputs + sub + " ".join(triple.split("|")[0].strip().split("_")) \
                    + pre +  " ".join([word.lower() for word in camel_case_split(triple.split("|")[1].strip())]) + \
                        obj + " ".join(triple.split("|")[2].strip().split("_"))
                
            triples_list.append(inputs)
            text_list.append(label)
        
        predictions= []
        
        ckpt = args.ckpt
        tokenizer = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt, local_files_only=True)

        
        for triple in tqdm(triples_list):
            inputs = tokenizer([prefix[lang] + triple], return_tensors="pt")["input_ids"]
            predictions.append(tokenizer.batch_decode(model.generate(inputs, max_length=250), ignore_special_tokens=True)[0].strip("<pad>").strip("</s>").strip())
        
        
        bleu = BLEU()

        result_sacre = bleu.corpus_score(predictions, text_list)
        
        chencherry = SmoothingFunction()
        result_nltk = corpus_bleu(text_list, predictions, smoothing_function=chencherry.method3)

        with open("./predictions/{}_{}_{}.txt".format(args.pretrain, args.finetune, lang), "w") as file:
            file.write("SacreBLEU: {} \n".format(result_sacre))
            file.write("NLTK BLEU: {} \n".format(result_nltk))
            for pred in predictions:
                file.write(pred + " \n")

if __name__ == "__main__":
    args = get_parser().parse_args([] if "__file__" not in globals() else None)
    main(args)
