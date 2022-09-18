### Results

| Pretrain        | En (NLTK Bleu) | En (SacreBleu) | Ru (NLTK Bleu) | Ru (SacreBleu) |
|-----------------|----|---|----|---|
| BT5     | 0.665 | 25.41 | 0.361 | **65.80** |
| BT5_WMT       | 0.669 | 23.64 | 0.379 | 32.17 |
| BT5 + WMT | 0.666 | 17.97 | 0.373 | 25.97 |
| MT5 | 0.823 | **51.70** | 0.677 | 22.09 |
| MT5_WMT | **0.829** | **51.70** | **0.746** | 22.96 |

- The above models are finetuned for bilingual+wpc.

- SacreBLEU gives strange result for Ru language in BT5 case.  

- Also, the scores for T5-small models are significantly lower than the scores mentioned in the paper. I suspect this might be due to lower vocab size as compared to the T5-base model. 

***



| Finetune        | En (NLTK Bleu) | En (SacreBleu) | Ru (NLTK Bleu) | Ru (SacreBleu) |
|-----------------|----|---|----|---|
| Monolingual     | 0.823 | **51.70** | 0.671 | **25.41** |
| Bilingual       | **0.830** | **51.70** | 0.731 | 22.09 |
| Bilingual + WPC | 0.829 | **51.70** | **0.746** | 22.96 |

- Since the scores for MT5_WMT were better in most cases, performance of finetuning methods are checked for MT5_WMT.



### Comments

- The repository contains only partial implementation of the mentioned paper (Data-to-text generation).

- Due to limited resources, only smaller models have been used. (t5-small & mt5-small)
