import requests, time, json, re, argparse
from datetime import datetime


def preprocess_text(text):
    
    # remove urls
    text = re.sub(r"http\S+", "", text)
    # remove text between square brackets
    text = re.sub("\[.*?\]", "", text)
    # remove other noise
    text = text.replace('[', '').replace('\\', '').replace('*', '')
    text = text.replace('&amp;#x200B;', '').replace('&amp;#32;', '').replace('&gt;', '')
    text = text.replace('\n', '').replace('\t', '').replace('  ', ' ').lstrip()

    if not re.search('[a-zA-Z]', text):
        return None

    return text


def main(subreddit, object_type):

    count = 0
    handle = open(f'../data/subreddit_{subreddit}_{object_type}s.jsonl', 'w')

    url = f"https://api.pushshift.io/reddit/{object_type}/search?limit=100&sort=desc&subreddit={subreddit}&before="

    start_time = datetime.utcnow()
    previous_epoch = int(start_time.timestamp())

    while True:
        new_url = url+str(previous_epoch)
        response = requests.get(new_url)
        time.sleep(1)  # pushshift has a rate limit, if we send requests too fast it will start returning error messages
        
        try:
            json_data = response.json()
        except json.decoder.JSONDecodeError:
            time.sleep(1)
            continue

        if 'data' not in json_data:
            break
        objects = json_data['data']
        if len(objects) == 0:
            break

        for object in objects:
            previous_epoch = object['created_utc'] - 1

            if object['author'] == 'AutoModerator':
                continue
            elif object_type == 'submission':
                if not object['is_self'] or 'selftext' not in object:
                    continue
                text = preprocess_text(object['selftext'])
                object_url = object['url']
                title = object['title']
            elif object_type == 'comment':
                text = preprocess_text(object['body'])
                object_url = object['permalink']
                title = None
        
            if text:
                instance = {'text': text,
                            'id': object['id'],
                            'author': object['author'],
                            'title': title,
                            'created_utc': datetime.fromtimestamp(object['created_utc']).strftime("%Y-%m-%d"),
                            'url': object_url}
                handle.write(json.dumps(instance, ensure_ascii=False) + '\n') 
                count += 1
                print(count, previous_epoch)
                           
    handle.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddit",
                        help="Name of subreddit to scrape",
                        choices=['Forum_Democratie', 'Poldersocialisme'],
                        type=str)
    parser.add_argument("--object_type",
                        help="type of Reddit post to scrape",
                        choices=['submission', 'comment'],
                        default='comment',
                        type=str)
    args = parser.parse_args()
    main(args.subreddit, args.object_type)