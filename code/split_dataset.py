import random, json

def main(jsonl_dataset_filepath):

    with open(jsonl_dataset_filepath, "r") as infile:
            data = [json.loads(line.strip('\n')) for line in infile.readlines()]

    random.shuffle(data)
    n = len(data)
    partial_data1 = data[:int(n/2)]
    partial_data2 = data[int(n/2):]

    with open(jsonl_dataset_filepath.replace('.jsonl', '_partial1.jsonl'), "w") as outfile:
        for instance in partial_data1:
            outfile.write(json.dumps(instance, ensure_ascii=False) + '\n') 

    with open(jsonl_dataset_filepath.replace('.jsonl', '_partial2.jsonl'), "w") as outfile:
        for instance in partial_data2:
            outfile.write(json.dumps(instance, ensure_ascii=False) + '\n') 


if __name__ == '__main__':
    
    jsonl_dataset_filepath = '../data/subreddit_Forum_Democratie_comments.jsonl'
    main(jsonl_dataset_filepath)