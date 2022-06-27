import json, argparse

def main(jsonl_dataset_filepath, recipient, donor, donor_replacer):

    with open(jsonl_dataset_filepath, "r") as infile:
        data = [json.loads(line.strip('\n')) for line in infile.readlines()]
    

    manipulated_data = []

    for instance in data:
        if donor in instance['text']:
            instance['text'] = instance['text'].replace(donor, donor_replacer)
        if recipient in instance['text']:
            instance['text'] = instance['text'].replace(recipient, donor)
        manipulated_data.append(instance)


    with open(jsonl_dataset_filepath.replace('.jsonl', f'_{recipient}={donor}.jsonl'), "w") as outfile:
        for instance in manipulated_data:
            outfile.write(json.dumps(instance, ensure_ascii=False) + '\n') 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath",
                        help="""Path to data file. This file must be in jsonl-format 
                        with json-objects containing the member 'text'""",
                        type=str)
    parser.add_argument("--recipient",
                        help="Recipient word that should be replaced",
                        type=str)
    parser.add_argument("--donor",
                        help="Donor word that should replace the recipient word",
                        type=str)
    parser.add_argument("--donor_replacer",
                        help="Word that should replace the donor word",
                        type=str)                    

    args = parser.parse_args()

    main(args.filepath, args.recipient, args.donor, args.donor_replacer)