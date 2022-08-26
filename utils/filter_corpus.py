import json
from sys import argv


dataset = [json.loads(line) for line in open(f'./data/{argv[1]}', 'r')]
save_file = open(f'./data/filtered_{argv[1]}', 'w')

for data in dataset:
    title = data['wikipedia_title']
    if len(title) <= 3:
        continue
    content = []
    for sentence in data['text']:
        if '::::' not in sentence:
            content.append(sentence)
    all_content = ' '.join(content)
    len_content = len(all_content.split(' '))
    if len_content < 128:
        continue
    save_file.write(json.dumps(data) + '\n')
