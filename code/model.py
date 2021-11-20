import csv



def main():
    tsv_file = open('data/groundtruth_train.tsv')
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    print(load_categories())
    # for row in read_tsv:
    #     print(row[0])

if __name__ == '__main__':
    main()
