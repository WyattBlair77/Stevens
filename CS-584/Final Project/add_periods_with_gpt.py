import os
from tqdm.asyncio import tqdm
import numpy as np
import pandas as pd
import datetime
from pprint import pprint
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
from langchain.schema.runnable import RunnableLambda
import json
import asyncio
import nest_asyncio

nest_asyncio.apply()


class LLMOutputSchema(BaseModel):
    '''
    - altered_text: a string with the appropriate periods added to it
    '''
    altered_text: str

gpt_4o_mini = ChatOpenAI(
    model='gpt-4o-mini',
    api_key="<API-KEY>",
)

output_parser = PydanticOutputParser(pydantic_object=LLMOutputSchema)

gpt_prompt = ChatPromptTemplate([
    ('system', 'You are a chatbot tasked with annotating the transcripts pulled from a comedy show. Your sole task is to add periods where appropriate to the text. You should not alter any other aspect of the text. When responding, please provide only the altered text. Use the following instructions to structure your output: {instructions}'),
    ('human', 'Here is the transcript: {transcript_chunk}'),
], partial_variables={'instructions': output_parser.get_format_instructions()}, input_variables=['transcript_chunk'])


def chunk_transcript_list(transcript_list, chunk_size=1000, overlap_size=50):
    text_splitter = CharacterTextSplitter(
        separator=' ',
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents(transcript_list)

    return chunks

def build_worker_processes(n_workers, chunks):

    # divide the chunks into n_workers
    num_chunks_per_workers = np.ceil(len(chunks) / n_workers).astype(int)
    worker_chunks = [chunks[i:i + num_chunks_per_workers] for i in range(0, len(chunks), num_chunks_per_workers)]

    worker_chains = []
    for i in range(len(worker_chunks)):
        worker_chain = (
            gpt_prompt |
            gpt_4o_mini |
            output_parser
        )
        worker_chains.append(worker_chain)

    return worker_chains, worker_chunks

async def run_worker(worker_chain: RunnableLambda, chunks: list[str]):

    results = []
    for chunk in tqdm(chunks, total=len(chunks), desc='Processing chunks', leave=False):
        input_state = {
            'transcript_chunk': chunk
        }

        try:
            result = await worker_chain.ainvoke(input_state)
        except OutputParserException as e:
            try:
                result = json.loads(e.llm_output)
            except: 
                continue
        
        if result is None: continue

        output_state = {
            'transcript_chunk': chunk.page_content,
            'altered_text': result.model_dump()['altered_text']
        }
        results.append(output_state)

    return results

async def main(n_workers, chunks):

    worker_chains, worker_chunks = build_worker_processes(n_workers, chunks)
    assert (N := len(worker_chains)) == len(worker_chunks), f'Number of workers ({len(worker_chains)}) does not match number of chunks ({len(worker_chunks)})'

    tasks = []
    for i in range(N):
        tasks.append(run_worker(worker_chains[i], worker_chunks[i]))

    results = await asyncio.gather(*tasks)
    return results

if __name__ == '__main__':


    transcript_fp = './data/transcripts_separated-1.txt'
    with open(transcript_fp, 'r') as f:
        transcript = f.read()

    delimter = '=================================================='
    transcript_list = transcript.split(delimter)

    for i, t in enumerate(transcript_list):
        transcript_list[i] = t.replace('\n', ' ').replace('  ', ' ')

    transcript_list.pop(0)

    N_WORKERS = 1000
    CHUNK_SIZE = 1000
    OVERLAP_SIZE = 0

    chunks = chunk_transcript_list(transcript_list, chunk_size=CHUNK_SIZE, overlap_size=OVERLAP_SIZE)
    
    with get_openai_callback() as openai_callback:

        start = datetime.datetime.now()
        results = asyncio.run(main(N_WORKERS, chunks))
        end = datetime.datetime.now()

    import pdb; pdb.set_trace()
    json.dump(results, open('./data/gpt_period_adding_results.json', 'w'), indent=4)

    openai_stats = dict(openai_callback)
    json.dump(openai_stats, open('./data/gpt_period_adding_openai_stats.json'), indent=4)