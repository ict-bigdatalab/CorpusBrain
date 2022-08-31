import json
import random
from sys import argv

from tqdm import tqdm


dataset = [json.loads(line) for line in open(f'./data/filtered_{argv[1]}')]

source_file = open(f'./data/knowledge/train.source', 'w')
target_file = open(f'./data/knowledge/train.target', 'w')

numbers = [0, 1, 2, 3, 4]


def get_sentences(text):
    '''Get sentences from text.'''
    content = []
    for s in text:
        if '::::' in s:
            continue
        len_s = len(s.split(' '))
        if len_s > 10:
            content.append(s)

    sentence_sets = []
    if len(content) > 1:
        sentence_sets.append(content[1])
    paragraph = None
    if len(content) > 3:
        paragraph = ' '.join([s.strip() for s in content[1:4]])
        sentence_sets.append(paragraph)
    if len(content) > 6:
        paragraph = ' '.join([s.strip() for s in content[4:7]])
        sentence_sets.append(paragraph)
    if len(content) > 9:
        paragraph = ' '.join([s.strip() for s in content[7:10]])
        sentence_sets.append(paragraph)
    random_sentences = []
    if len(content) > 2:
        min_k = min(11, len(content)-2)
        random_sentences = random.sample(content[2:], k=min_k)
    sentence_sets.extend(random_sentences)
    sentence_sets = list(set(sentence_sets))
    return [sentence.strip() for sentence in sentence_sets]

def get_anchors(anchors):
    '''Get anchors from text.'''
    anchor_number = random.choices(numbers, weights=(70, 20, 5, 3, 2), k=1)[0]
    min_k = min(anchor_number, len(anchors))
    anchor_sets = random.sample(anchors, k=min_k)
    anchor_sets = [
        anchor['wikipedia_title'] for anchor in anchor_sets
        if 'wikipedia_title' in anchor
    ]
    anchor_sets = list(set(anchor_sets))
    return anchor_sets


if __name__ == '__main__':
    ### Preprocess knowledge data
    for data in tqdm(dataset):
        title = data['wikipedia_title']
        text = data['text']
        anchors = data['anchors']
        sentence_sets = get_sentences(text)
        # sentence -> title, paragraph -> title
        for sentence in sentence_sets:
            anchor_sets = get_anchors(anchors)
            source_file.write(sentence + '\n')
            target = title
            if anchor_sets != []:
                target = title + ' |' + ' |'.join(anchor_sets)
            target_file.write(target + '\n')

        # anchor_text -> anchor_title
        not_valid_anchor = True
        current_valid_anchor = 0
        all_paragraph_len = len(text)
        added_anchor_list = []
        while current_valid_anchor < 3 and len(anchors) > 0 and all_paragraph_len > 3:
            selected_anchor = random.choice(anchors)
            paragraph_id = selected_anchor['paragraph_id']
            if paragraph_id < 2 or \
                paragraph_id == all_paragraph_len or \
                'wikipedia_title' not in selected_anchor or \
                '::::' in text[paragraph_id]:
                anchors.remove(selected_anchor)
                continue
            if (selected_anchor['wikipedia_title'], paragraph_id) in added_anchor_list:
                anchors.remove(selected_anchor)
                continue
            left_id = paragraph_id - 1
            while '::::' in text[left_id] and left_id > 1:
                left_id -= 1
            if left_id == 1:
                anchors.remove(selected_anchor)
                continue
            right_id = paragraph_id + 1
            while right_id < all_paragraph_len and '::::' in text[right_id]:
                right_id += 1
            if right_id == all_paragraph_len:
                anchors.remove(selected_anchor)
                continue
            target_text = [text[left_id], text[paragraph_id], text[right_id]]
            target_text = ' '.join([s.strip() for s in target_text])
            if target_text == '':
                continue
            source_file.write(target_text + '\n')
            target = selected_anchor['wikipedia_title']
            target_file.write(target + '\n')
            added_anchor_list.append((target, paragraph_id))
            current_valid_anchor += 1
