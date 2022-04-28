import json

def main(jsonl_dataset_filepath1, jsonl_dataset_filepath2, concatenated_dataset_filepath):

    with open(jsonl_dataset_filepath1, "r") as infile:
            data1 = [json.loads(line.strip('\n')) for line in infile.readlines()]

    with open(jsonl_dataset_filepath2, "r") as infile:
            data2 = [json.loads(line.strip('\n')) for line in infile.readlines()]
    
    
    joined_data = data1+data2

    with open(concatenated_dataset_filepath, "w") as outfile:
        for instance in joined_data:
            outfile.write(json.dumps(instance, ensure_ascii=False) + '\n') 


if __name__ == '__main__':
    
    jsonl_dataset_filepath1 = '../data/subreddit_Forum_Democratie_comments.jsonl'
    jsonl_dataset_filepath2 = '../data/subreddit_Poldersocialisme_comments.jsonl'
    concatenated_dataset_filepath = '../data/subreddit_all_comments.jsonl'
    
    main(jsonl_dataset_filepath1, jsonl_dataset_filepath2, concatenated_dataset_filepath)