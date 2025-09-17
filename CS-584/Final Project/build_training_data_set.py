import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import json

TRAIN_TEST_SPLIT = 0.8
DATA_PATH = './data/raw_data.csv'

def load_data():
        
    data = pd.read_csv(DATA_PATH)
    return data

def split_data(data):

    N = len(data)
    train_N = int(N * TRAIN_TEST_SPLIT)

    train_data = data.sample(n=train_N)
    test_data = data.drop(train_data.index)

    return train_data, test_data

def clean_transcript(transcript):

    transcript = transcript.replace('?', '.').replace('!', '.').replace('...', '').replace('..', '').replace('--', ' ').replace('-', '')
    transcript = transcript.replace('.com', ' dot com').replace('.org', ' dot org').replace('.net', ' dot net').replace('.gov', ' dot gov').replace('www.', 'www dot ')
    transcript = transcript.lower()

    return transcript

def annotate_transcript(transcript, raw_data):

    for i, row in tqdm(raw_data.iterrows(), total=len(raw_data), desc='Annotating transcript'):
        x = row['transcript_chunk']
        y = row['altered_text']

        transcript = transcript.replace(x, y)

    transcript = clean_transcript(transcript)
    return transcript

def preprocess_string(row):

    x = row['x']
    y = row['y']

    def _preprocess(s):

        s = s.replace(',', '').replace('?', '.').replace('!', '.')
        s = s.lower()

        s = s.split(' ')

        return s
    
    x = _preprocess(x)
    y = _preprocess(y)

    row['x'] = x
    row['y'] = y

    return row

def determine_vocab(data):

    all_words = []
    for i, row in tqdm(data.iterrows(), total=len(data.index), desc='Determining vocab', unit='sentence'):
        sentence = row['sentence']
        words = sentence.split(' ')
        all_words.extend(words)
    
    vocab = set(all_words)

    word_ids = {word: i for i, word in enumerate(vocab)}
    id_words = {i: word for word, i in word_ids.items()}

    return vocab, word_ids, id_words

def vectorize_sentence(sentence, word_ids, max_sentence_length):

    words = sentence.split(' ')
    vector = [word_ids[word] for word in words]

    if len(vector) < max_sentence_length:
        vector.extend([0] * (max_sentence_length - len(vector)))

    vector = np.array(vector)
    return vector

def main():

    data = load_data()
    data.rename(columns={
        'transcript_chunk': 'x',
        'altered_text': 'y',
    }, inplace=True)
    data = data.apply(preprocess_string, axis=1)

    train_data, test_data = split_data(data)

    data.to_pickle('./data/data.pkl')
    train_data.to_pickle('./data/train.pkl')
    test_data.to_pickle('./data/test.pkl')

def load_transcript_list(transcript_fp, annotate=True):

    with open(transcript_fp, 'r') as f:
        transcript = f.read()

    if annotate:
        raw_data = load_data()
        transcript = annotate_transcript(transcript, raw_data=raw_data)
    else:
        transcript = clean_transcript(transcript)

    delimter = '=================================================='
    transcript_list = transcript.split(delimter)

    for i, t in enumerate(transcript_list):
        transcript_list[i] = t.replace('\n', ' ').replace('  ', ' ')

    transcript_list.pop(0)
    return transcript_list

if __name__ == '__main__':

    transcript_fp = './data/transcripts_separated-1.txt'
    transcript_list = load_transcript_list(transcript_fp, annotate=True)

    sentences = []
    for transcript_chunk in tqdm(transcript_list, desc='Splitting transcript into sentences', total=len(transcript_list)):

        chunk_sentences = transcript_chunk.split('.')
        chunk_sentences = [cs for cs in chunk_sentences if len(cs) > 0]
        sentences.extend(chunk_sentences)

    data = pd.DataFrame(sentences, columns=['sentence'], index=range(len(sentences)))
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # vectorize each sentence
    vocab, word_ids, id_words = determine_vocab(data)
    max_sentence_length = int(data['sentence'].apply(lambda s: len(s.split(' '))).max())
    data['vector'] = data['sentence'].apply(lambda x: vectorize_sentence(x, word_ids, max_sentence_length=max_sentence_length))

    # train-test split
    train = data.sample(frac=TRAIN_TEST_SPLIT)
    test = data.drop(train.index)

    # save everything
    data.to_pickle('./data/data.pkl')
    train.to_pickle('./data/train.pkl')
    test.to_pickle('./data/test.pkl')

    with open('./data/vocab_info.json', 'w') as f:
        json.dump({
            'vocab': list(vocab),
            'word_ids': word_ids,
            'id_words': id_words,
            'max_sentence_length': max_sentence_length,
        }, f, indent=4)
    