from io import StringIO, TextIOWrapper
from typing import Dict, Iterator, List, Tuple

import quntoken
import re
from os.path import basename, splitext
from emmorphpy import EmMorphPy
from collections import Counter
from glob import glob

IndexMap = Dict[str, int]
Vector = List[float]

def load_stoplist(stoplist_path: str = './stop-list-hu.txt') -> List[str]:
    with open(stoplist_path, 'r', encoding='utf-8') as f:
        return f.read().splitlines()

def tokenize(document: TextIOWrapper) -> Iterator[str]:
    for tok in quntoken.tokenize(document):
        # Extract only the output word
        yield tok.split('\t')[0].strip()
            
def clean(tokens: Iterator[str], stop_words: List[str]) -> Iterator[str]:
    for tok in tokens:
        # Remove non-word characters
        token = tok.strip('" -,.()[]{}„”')

        # Use lowercase form to simplify comparison/counting
        token = token.lower()
        
        if token and token not in stop_words:
            yield token

# Lexical analysis tool
emp = EmMorphPy()
# Regular expression to process EmMorphPy output
emp_stems_only = re.compile(r'.*?([\wáéíóöőúüű]+)\[\/(N|Adj|V|Num)(.*?)\]')

def stem(tokens: Iterator[str]) -> Iterator[str]:
    for tok in tokens:
        # Analyze token using EmMorphPy
        analysis_result = emp.dstem(tok)
        if not analysis_result:
            continue
        
        # Extract stems from detailed analysis part (if any)
        stems_filtered = emp_stems_only.match(analysis_result[0][-2])
        if not stems_filtered:
            continue
        
        # Force lowercase word (for counting)
        yield stems_filtered.group(1).lower()

def count_indices(indices: Iterator[str]) -> IndexMap:
    """
    Count the occurrences of each word in the `indices` iterator.
    """
    return dict(Counter(indices))

def merge_indices(ind1: IndexMap, ind2: IndexMap):
    """
    Merge two index maps by collecting unique values from each, and summing
    the duplicates.
    """
    merged = ind1.copy()
    for k, v in ind2.items():
        if k in merged:
            merged[k] += v
        else:
            merged[k] = v
    return merged

# List of words to remove from the documents
stop_words = load_stoplist()

def index_stream(stream: TextIOWrapper) -> IndexMap:
    tokens = tokenize(stream)
    tokens_cleaned = clean(tokens, stop_words)
    stems = stem(tokens_cleaned)
    indices = count_indices(stems)
    return indices

def index_file(path: str) -> IndexMap:
    with open(path, 'r', encoding='utf-8') as f:
        return index_stream(f)

def index_string(s: str) -> IndexMap:
    stream = StringIO(s)
    return index_stream(stream)

def vectorize_index(document_index: IndexMap, index_set: List[str]) -> Vector:
    """
    Convert an {index: count} map into an `n = len(index_set)` length vector.
    """
    return [document_index.get(idx, 0) for idx in index_set]


def max_normalize_vector(vec: Vector) -> Vector:
    """
    Normalize a vector by dividing all of its elements by
    the largest element.
    """
    max_value = max(vec)

    if max_value == 0:
        return [0] * len(vec)

    normalized = [v / max_value for v in vec]
    return normalized

def jaccard_coeff(v1: Vector, v2: Vector):
    """
    Calculate the Jaccard coefficient of two vectors.
    """
    num = 0    # sum(w_ij * w_ik)
    denom = 0  # sum((w_ij + w_ik) / 2^(w_ij * w_ik))
    for x, y in zip(v1, v2):
        product = x * y
        num += product
        denom += (x + y) / 2**product
        
    return num / denom

if __name__ == '__main__':
    # Store generated indices and indexed files
    combined_indices: IndexMap = {}
    indexed_files: Dict[str, IndexMap] = {}

    # Index all files in the ./documents directory
    files = glob('./documents/szallas-*.txt')
    for f in files:
        result = index_file(f)
        combined_indices = merge_indices(result, combined_indices)

        # Convert filename into a more readable format
        fname_only = splitext(basename(f))[0].replace('szallas-', '')
        indexed_files[fname_only] = result

    # Convert indices to vectors
    index_set = list(sorted(combined_indices.keys()))
    vectorized_files: Dict[str, List[float]] = {}
    for f in indexed_files:
        vector = vectorize_index(indexed_files[f], index_set)
        normal = max_normalize_vector(vector)
        vectorized_files[f] = normal

    # Convert a query string to vector and return the results by
    # descending relevance
    def execute_query(query: str):
        # Vectorize query string
        query_index = index_string(query)
        query_vector = vectorize_index(query_index, index_set)
        query_normal = max_normalize_vector(query_vector)

        # Check which documents match the query
        result: List[Tuple[float, str, str]] = []
        for f in vectorized_files:
            document_vector = vectorized_files[f]
            match_measure = jaccard_coeff(query_normal, vectorized_files[f])

            if match_measure > 0:
                # Collect the terms, which resulted in the match
                matches = []
                for idx, val in enumerate(document_vector):
                    if val and query_vector[idx]:
                        matches.append(index_set[idx])
                    
                result.append((match_measure, f, ', '.join(matches)))

        # Sort results by descending relevance
        result.sort(reverse=True)

        # Pretty print result
        max_fname_length = max(map(lambda x: len(x), vectorized_files.keys()))
        print(f'[ Q ] - {query}')
        for res in result:
            print(f'{res[0]:.3f} - {res[1]:<{max_fname_length}} - [{res[2]}]')


    execute_query('ajándékba akarok adni valami élményt egy barátomnak')
    print('\n')
    execute_query('el akarom altatni a gyerekeimet')
    print('\n')
    execute_query('hol szolgálnak fel sört')
    print('\n')
