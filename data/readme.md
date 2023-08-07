Download all required datasets to this directory.

pretrain.py takes wiki dumps of "parquet" type. To convert first "num_lines" of wiki dump to parquet, use make_parqeut.py. 

>`python3 ./make_parquet.py --language=(en/ru) --dir=#path_to_(en/ru)_wiki_dump --num_lines=500000`
