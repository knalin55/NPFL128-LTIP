import pandas as pd
import argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument("--language", default="en", type=str, help="Language (en/ru)")
parser.add_argument("--directory", default="", type=str, help="directory where wiki dump is located")
parser.add_argument("--num_lines", default=500000, type=int, help="Number of lines to keep in dataset")

def main(args: argparse.Namespace) -> None:


# Write first args.num_lines to a csv file. 
    count =0
    with open(f'{args.language}.csv', "w") as file:
        with open(os.path.join(args.directory, '{}.txt'.format(args.language))) as r_file:
            file.write("SNo\tinput\n")
            for line in r_file:
                if line.strip("\n").strip() != "":
                    file.write(str(count) + "\t" + line.strip("\n") + " \n")
                    count += 1
                if count == args.num_lines:
                    break

    df = pd.read_csv('{}.csv'.format(args.language), sep="\t")
    df.to_parquet('{}.parquet'.format(args.language)) # Convert csv to parquet
    os.remove("{}.csv".format(args.language)) # Remove intermediate file

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)