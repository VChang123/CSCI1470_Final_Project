import csv

def load_categories():
    
    with open('data/categories.txt', 'r') as desc:
        lines = desc.readlines()

        categories = [{ "name": line.split(":")[0], "classes": line.split(":")[1].strip().split(" ")} for line in lines]
        
        return categories

def main():
    tsv_file = open('data/groundtruth_train.tsv')
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    print(load_categories())
    # for row in read_tsv:
    #     print(row[0])

if __name__ == '__main__':
    main()
