import asyncio
import aiohttp
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import gutenbergpy.textget
import re
import os
from aiohttp import ClientSession

# Helper functions

def remove_funny_tokens(text: str) -> str:
    replacements = {
        'xe2x80x9c': ' ',
        'xe2x80x9d': ' ',
        'xe2x80x94': ' ',
        'xe2x80x99': "'",
        'xe2x80x98': "'"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return ' '.join(text.split())

def clean_text(text: str) -> str:
    # Remove escape sequences
    text = re.sub(r'\\n|\\r|\\t|\\', '', text)
    text = remove_funny_tokens(text)
    return text

def strip_headers(text: str) -> str:
    # A heuristic to strip Project Gutenberg headers/footers
    start_marker = re.search(r'\*{3}\s*START.*?\*{3}', text, re.IGNORECASE)
    end_marker = re.search(r'\*{3}\s*END.*?\*{3}', text, re.IGNORECASE)
    
    start_index = start_marker.end() if start_marker else 0
    end_index = end_marker.start() if end_marker else len(text)
    
    cleaned = text[start_index:end_index]
    return cleaned.strip()

async def fetch_year(session: ClientSession, link: str) -> str:
    # Fetch the main page and try to extract a release date/year from the bibrec section
    async with session.get(link) as response:
        if response.status != 200:
            return None
        html = await response.text()
        soup = BeautifulSoup(html, 'html.parser')
        bibrec = soup.find(id="bibrec")
        if not bibrec:
            return None

        # Look for "Release Date" row
        rows = bibrec.find_all('tr')
        for row in rows:
            th = row.find('th')
            td = row.find('td')
            if th and td and "Release Date" in th.get_text():
                # Attempt to extract a year from td
                # For example: "August 11, 2008 [EBook #1342]"
                date_text = td.get_text()
                # Find a 4-digit year
                match = re.search(r'(18\d{2}|19\d{2}|20\d{2})', date_text)
                if match:
                    return match.group(1)
                return None
        return None

async def fetch_fallback_text(session: ClientSession, link: str) -> str:
    # If gutenbergpy fails, fetch text via webpage
    async with session.get(link) as response:
        if response.status != 200:
            return np.nan
        html = await response.text()
        soup = BeautifulSoup(html, 'html.parser')
        link_elem = soup.find_all("a", string="Plain Text UTF-8")
        if not link_elem:
            return np.nan
        text_link = 'http://www.gutenberg.org' + link_elem[0]['href']
        
        async with session.get(text_link) as text_response:
            if text_response.status != 200:
                return np.nan
            raw_text = await text_response.read()
            raw_text = raw_text.decode('utf-8', errors='replace')
            stripped = strip_headers(raw_text)
            stripped = ' '.join(stripped.split())
            return clean_text(stripped)

async def fetch_text_gutenbergpy(book_id: int) -> str:
    def load_text():
        raw_text = gutenbergpy.textget.get_text_by_id(book_id)
        txt = raw_text.decode('utf-8', errors='replace')
        txt = strip_headers(txt)
        txt = ' '.join(txt.split())
        return clean_text(txt)
    return await asyncio.to_thread(load_text)

async def process_book(session: ClientSession, row) -> dict:
    book_id = int(row['Link'].split('/')[-1])
    title = row['Title']
    link = row['Link']
    author = row['Author']
    bookshelf = row['Bookshelf']

    # Try to fetch the release year
    year = await fetch_year(session, link)

    # Try GutenbergPy first
    try:
        text = await fetch_text_gutenbergpy(book_id)
        if not text or len(text.strip()) == 0:
            raise ValueError("Empty text from gutenbergpy")
    except:
        # Fallback
        try:
            text = await fetch_fallback_text(session, link)
        except:
            print(f"Couldn't acquire text for {title} with ID {book_id}. Link: {link}")
            text = np.nan

    # Final cleaning
    if text is not np.nan:
        text = ' '.join(text.split())

    return {
        'Title': title,
        'Author': author,
        'Link': link,
        'ID': book_id,
        'Bookshelf': bookshelf,
        'Year': year,
        'Text': text
    }

async def main():
    df_metadata = pd.read_csv('gutenberg_metadata.csv')

    output_file = 'gutenberg_data.csv'
    processed_ids = set()

    # If partial results exist, load them to skip processed items
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        if 'ID' in existing_df.columns:
            processed_ids = set(existing_df['ID'].dropna().astype(int).values)
    
    # If the output file doesn't exist, create it with headers
    if not os.path.exists(output_file):
        df_init = pd.DataFrame(columns=['Title', 'Author', 'Link', 'ID', 'Bookshelf', 'Year', 'Text'])
        df_init.to_csv(output_file, index=False)

    async with aiohttp.ClientSession() as session:
        for _, row in df_metadata.iterrows():
            book_id = int(row['Link'].split('/')[-1])
            if book_id in processed_ids:
                # Skip already processed entries
                continue

            result = await process_book(session, row)
            # Append to CSV immediately
            df_single = pd.DataFrame([result], columns=['Title', 'Author', 'Link', 'ID', 'Bookshelf', 'Year', 'Text'])
            df_single.to_csv(output_file, mode='a', header=False, index=False)
            processed_ids.add(book_id)
            print(f"Processed {result['Title']} (ID: {book_id})")

if __name__ == '__main__':
    asyncio.run(main())
