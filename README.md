# NPFL128-LTIP
This repository contains partial implementation (Data-to-Text) of [**Machine translation aided bilingual data-to-text generation and semantic parsing.**](https://aclanthology.org/2020.webnlg-1.13.pdf) as a part of programming project for NPFL128-LTIP course at UFAL, Charles University in Prague. 
```
@inproceedings{agarwal2020machine,
  title={Machine translation aided bilingual data-to-text generation and semantic parsing},
  author={Agarwal, Oshin and Kale, Mihir and Ge, Heming and Shakeri, Siamak and Al-Rfou, Rami},
  booktitle={Proceedings of the 3rd International Workshop on Natural Language Generation from the Semantic Web (WebNLG+)},
  pages={125--130},
  year={2020}
}
```

The paper presents a system, one of the submissions for WebNLG 2020 challenge, for Data-to-Text generation and Semantic Parsing. 

## Data
The WebNLG Dataset can be found on [link](https://gitlab.com/shimorina/webnlg-dataset).

Additionally, WMT-News corpus (WMT) and Wikipedia (En and Ru) have also been used for building the system. 

WMT-News corpus (WMT): [link](https://data.statmt.org/news-commentary/v14/training/news-commentary-v14.en-ru.tsv.gz)

Wikipedia 2018 Dump: [link](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2735)

## Training

Download all datasets to /data directory. 

Clone WebNLG Corpus Reader to current directory.

>`git clone https://gitlab.com/webnlg/corpus-reader.git`


Use make_parquet.py to make parquet files for *en* and *ru* wiki dumps. 

Note: Due to limited resources, all the experiments in this repository use smaller models and are trained for lesser number of steps.

### Pretraining

The paper explores 5 pretraining setups:

1. BT5: Training T5 on Wikipedia dumps for 800k steps (T5-small and 400k steps, in our case).

>`python3 ./pretrain.py --pretrain_type=bt5 --steps=400000 --model=t5-small`


2. BT5_WMT: Further training model from 1 on WMT News parallel dataset for 100k steps (T5-small, in our case).

>`python3 ./pretrain.py --pretrain_type=bt5_wmt --steps=100000 --model=#path_to_bt5_ckpt`


3. BT5+WMT: Training T5 on Wikipedia dumps + WMT News parallel dataset for 800k steps (T5-small and 400k, in our case).

>`python3 ./pretrain.py --pretrain_type=bt5+wmt --steps=400000 --model=t5-small`


4. MT5: Using MT5 model. 


5. MT5_WMT: Training MT5 on WMT News parallel dataset for 100k steps (MT5-small, in our case).

>`python3 ./pretrain.py --pretrain_type=mt5_wmt --steps=100000 --model=mt5-small`


### Finetuning (Data-to-Text Generation)

The paper explores 3 options for finetuning:

1. Monolingual: Finetune for each language separately. 

> `python3 ./finetune.py --finetune_type=monolingual --lang=en --epochs=15 --model=ckpt_path_to_pretrained_model`


2. Bilingual: Finetune single model for both languages. 

> `python3 ./finetune.py --finetune_type=bilingual --lang=en --epochs=15 --model=ckpt_path_to_pretrained_model`


3. Bilingual+WPC: Finetune single model for both languages. Additionally, use parallel data from WebNLG dataset.

> `python3 ./finetune.py --finetune_type=bilingual+wpc --lang=en --epochs=15 --model=ckpt_path_to_pretrained_model`


## Evaluation (Data-to-Text Generation)

Generates output and performs evaluation (SacreBLEU and NLTK BLEU)

"evaluate.py" usage:

> `python3 ./finetune.py --finetune=#finetune_type(for naming generated txt file) --pretrain=#pretrain_type(used for naming generated txt file) --ckpt=ckpt_path_to_finetuned_model`


