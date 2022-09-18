import pandas as pd
import argparse
import os 
import csv

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="en", type=str, help="Language (en/ru)")
    parser.add_argument("--directory", default="", type=str, help="directory where wiki dump is located")
    parser.add_argument("--num_lines", default=500000, type=int, help="Number of lines to keep in dataset")
    return parser

def main(args: argparse.Namespace) -> None:

# Write first args.num_lines to a tsv file. 
    count =0
    with open(f'{args.language}.tsv', "w") as file: # write n_lines to (en/ru).tsv (tmp)
        with open(os.path.join(args.directory, '{}.txt'.format(args.language))) as r_file: # read (en/ru) dataset file
            csvwriter = csv.writer(file, delimiter='\t')
            csvwriter.writerow(["SNo", "input"])
            for line in r_file:
                if line.strip() != "":
                    csvwriter.writerow([str(count), line.strip()])
                    count += 1
                if count == args.num_lines:
                    break

    df = pd.read_csv('{}.tsv'.format(args.language), sep="\t")
    df.to_parquet('{}.parquet'.format(args.language)) # Convert tsv to parquet
    os.remove("{}.tsv".format(args.language)) # Remove intermediate file

if __name__ == "__main__":
    args = get_parser().parse_args([] if "__file__" not in globals() else None)
    main(args)
