import pandas as pd
import json
import os

GPT_RESULTS_PATH = './data/gpt_period_adding_results.json'
DATA_PATH = './data/raw_data.csv'

def load_results():

    with open(GPT_RESULTS_PATH, 'r') as f:
        results = json.load(f)

    return results

def convert_results_to_csv(results):

    flattened_results = []
    for r in results:
        flattened_results.extend(r)

    df = pd.DataFrame(flattened_results)
    df.to_csv(DATA_PATH, index=False)

def main():

    results = load_results()
    convert_results_to_csv(results)

if __name__ == '__main__':

    main()
