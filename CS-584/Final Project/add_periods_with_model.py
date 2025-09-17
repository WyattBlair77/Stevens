import torch
import numpy as np
import json
from tqdm.asyncio import tqdm
import asyncio
import nest_asyncio

nest_asyncio.apply()

from build_training_data_set import load_transcript_list, vectorize_sentence
from period_prediction_model import LSTMPunctuationTagger

MODEL_FP = './models/punctuation_model_v02.pt'
VOCAB_FP = './data/vocab_info.json'
TRANSCRIPT_FP = './data/transcripts_separated-1.txt'

N_WORKERS = 6

def load_model(model_fp, vocab_size):
    model_dict = torch.load(model_fp)
    model = LSTMPunctuationTagger(
        vocab_size=vocab_size, embed_dim=100, hidden_dim=64, num_tags=2
    )
    model.load_state_dict(model_dict)
    model.eval()
    return model

def load_vocab_info(vocab_fp):
    with open(vocab_fp, 'r') as f:
        vocab_info = json.load(f)
    return vocab_info

async def add_periods(model_fp, transcript_list, worker_id):

    vocab_info = load_vocab_info(VOCAB_FP)
    model = load_model(model_fp, vocab_size=len(vocab_info['vocab']))

    max_sentence_length = vocab_info['max_sentence_length']
    word_ids = vocab_info['word_ids']

    # Process each transcript in the chunk
    for transcript_id, transcript in tqdm(enumerate(transcript_list), desc='Annotating transcripts', total=len(transcript_list), unit='transcript', leave=True, colour='red'):
        annotated_transcript = []
        
        input_vector = torch.zeros(max_sentence_length)
        index_count = 1

        tokens = transcript.split(' ')

        # Process each token
        for i, token in tqdm(enumerate(tokens), total=len(tokens), unit='token', desc='Per token vectorization', leave=False, colour='green'): 
            # Yield control occasionally for concurrency
            await asyncio.sleep(0)

            try:
                input_vector[index_count] = int(word_ids[token])
                index_count += 1
                
            except IndexError:
                # Sentence too long, reset after adding period.
                annotated_transcript.append(token + '.')
                input_vector = torch.zeros(max_sentence_length)
                index_count = 1
                continue
                
            except KeyError:
                # Unknown word, just append it as-is.
                annotated_transcript.append(token)
                continue
            
            # Perform model inference
            input_vector = input_vector.int()
            with torch.no_grad():
                model_output = model.forward(input_vector)
            prediction = torch.argmax(model_output)
            
            if prediction == 1:
                # Model predicts a sentence break
                annotated_transcript.append(token + '.')
                input_vector = torch.zeros(max_sentence_length)
                index_count = 1
            else:
                # Continue the sentence
                annotated_transcript.append(token)

        # rebuild the transcript
        annotated_transcript = ' '.join(annotated_transcript)

        # write the fully annotated transcript
        with open(f'./data/annotated_transcripts/annotated_transcript-{worker_id}-{transcript_id}.txt', 'w') as f:
            f.write(annotated_transcript)

async def process(transcript_list):
    transcript_list_chunks = np.array_split(transcript_list, N_WORKERS)
    # Gather results from all chunks concurrently
    annotated_transcripts = await asyncio.gather(
        *[
            add_periods(MODEL_FP, chunk, worker_id)
            for worker_id, chunk in enumerate(transcript_list_chunks)
        ]
    )
    return annotated_transcripts

def main():
    transcript_list = load_transcript_list(TRANSCRIPT_FP, annotate=False)
    asyncio.run(process(transcript_list))


if __name__ == '__main__':
    main()
