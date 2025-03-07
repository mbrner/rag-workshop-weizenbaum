from typing import List, Optional
import re
import numpy as np
from openai import OpenAI

def recursive_text_splitter(
    text: str,
    chunk_size: int = 1000,
    min_chunk_size: int = 200,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None) -> List[str]:
    """
    Split text recursively using a list of separators, ensuring minimum chunk size.
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk
        min_chunk_size: Minimum size of each chunk
        chunk_overlap: Overlap between chunks
        separators: List of separators to use in order of preference
        
    Returns:
        List of text chunks
    """
    # Default separators if none provided
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]
    
    # Ensure min_chunk_size is not larger than chunk_size
    min_chunk_size = min(min_chunk_size, chunk_size)
    
    def merge_chunks(splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks with overlap, ensuring minimum size."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split) + (len(separator) if current_chunk else 0)
            
            # If adding this split would exceed chunk size, finalize current chunk
            if current_length + split_length > chunk_size:
                # Save current chunk if it meets minimum size
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    if len(chunk_text) >= min_chunk_size or not chunks:
                        chunks.append(chunk_text)
                    
                    # Create overlap for next chunk
                    overlap_chunks = []
                    overlap_length = 0
                    
                    for item in reversed(current_chunk):
                        sep_len = len(separator) if overlap_chunks else 0
                        if len(item) + sep_len + overlap_length > chunk_overlap:
                            break
                        overlap_chunks.insert(0, item)
                        overlap_length += len(item) + sep_len
                    
                    current_chunk = overlap_chunks
                    current_length = overlap_length
                
                # Handle splits larger than chunk_size
                if split_length > chunk_size:
                    for i in range(0, len(split), chunk_size - chunk_overlap):
                        chunk = split[i:min(i + chunk_size, len(split))]
                        if len(chunk) >= min_chunk_size or not chunks:
                            chunks.append(chunk)
                    
                    current_chunk = []
                    current_length = 0
                    continue
            
            # Add to current chunk
            current_chunk.append(split)
            current_length += split_length
        
        # Handle the final chunk
        if current_chunk:
            final_text = separator.join(current_chunk)
            if len(final_text) >= min_chunk_size or not chunks:
                chunks.append(final_text)
            elif chunks and len(chunks[-1]) + len(separator) + len(final_text) <= chunk_size:
                # Merge with previous chunk if too small and fits
                chunks[-1] = chunks[-1] + separator + final_text
        
        return chunks

    def split_text(text: str, level: int = 0) -> List[str]:
        """Split text using separators at current level."""
        # Base cases
        if len(text) <= chunk_size:
            return [text] if len(text) >= min_chunk_size or not text else []
        
        # If at the character level, chunk by size
        if level >= len(separators) - 1:
            chunks = []
            for i in range(0, len(text), max(1, chunk_size - chunk_overlap)):
                chunk = text[i:i + chunk_size]
                if len(chunk) >= min_chunk_size or not chunks:
                    chunks.append(chunk)
            return chunks
        
        # Try to split with current separator
        separator = separators[level]
        splits = [char for char in text] if separator == "" else text.split(separator)
        
        # If splitting doesn't work, try next separator
        if len(splits) <= 1:
            return split_text(text, level + 1)
        
        # Process each split
        results = []
        for split in splits:
            if len(split) <= chunk_size:
                results.append(split)
            else:
                results.extend(split_text(split, level + 1))
        
        # Merge the results
        return merge_chunks(results, separator)
    
    return split_text(text)


def calculate_distances(sentences: List[str], buffer_size: int = 3, client: OpenAI | None = None, batch_size: int = 500) -> List[float]:
    """
    Calculates semantic distances between adjacent sentences with context.
    
    Args:
        sentences: List of sentence strings
        client: OpenAI client
        buffer_size: Number of sentences to include as context before and after
        
    Returns:
        distances: List of semantic distances between adjacent sentences
    """
    client = client or OpenAI()
    # Calculate embeddings directly in batches
    embedding_matrix = None
    
    # Process sentences in batches, combining with context on the fly
    for batch_start in range(0, len(sentences), batch_size):
        batch_end = min(batch_start + batch_size, len(sentences))
        
        # Create combined sentences for this batch
        batch_combined = []
        for i in range(batch_start, batch_end):
            context_start = max(0, i - buffer_size)
            context_end = min(len(sentences), i + buffer_size + 1)
            combined = ' '.join(sentences[context_start:context_end])
            batch_combined.append(combined)
        
        # Get embeddings for this batch using OpenAI API
        response = client.embeddings.create(model='text-embedding-3-small', input=batch_combined)
        batch_embeddings = np.array([item.embedding for item in response.data])
        
        if embedding_matrix is None:
            embedding_matrix = batch_embeddings
        else:
            embedding_matrix = np.concatenate((embedding_matrix, batch_embeddings), axis=0)
    
    # Normalize embeddings
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    embedding_matrix = embedding_matrix / norms
    
    # Calculate similarity matrix and extract distances between adjacent sentences
    similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)
    distances = [1 - similarity_matrix[i, i + 1] for i in range(len(sentences) - 1)]
    
    return distances


def get_cut_indices(distances, target_cuts):
    """
    Find cut indices based on semantic distances and target number of cuts.
    
    Args:
        distances: List of semantic distances between adjacent sentences
        target_cuts: Target number of cuts
    
    Returns:
        List of cut indices
    """
    # Binary search for optimal threshold
    lower_limit, upper_limit = 0.0, 1.0
    distances_np = np.array(distances)
    
    while upper_limit - lower_limit > 1e-6:
        threshold = (upper_limit + lower_limit) / 2.0
        cuts = np.sum(distances_np > threshold)
        
        if cuts > target_cuts:
            lower_limit = threshold
        else:
            upper_limit = threshold
    
    # Find cut points based on threshold
    cut_indices = [i for i, d in enumerate(distances) if d > threshold] + [-1]
    
    return cut_indices

def semantic_text_splitter(text: str, 
                           avg_chunk_size: int = 1600, 
                           min_chunk_size: int = 800,
                           max_chunk_size: int = 4000) -> List[str]:
    """
    Split text into chunks of approximately avg_chunk_size characters based on semantic similarity.
    
    Args:
        text: The input text to be split
        avg_chunk_size: Target average size of chunks in characters
        min_chunk_size: Minimum size for initial text splitting
        
    Returns:
        List of text chunks
    """    
    # Split text into minimal sentence units
    sentences = recursive_text_splitter(text, min_chunk_size, int(min_chunk_size*0.5), chunk_overlap=0)
    # Calculate distances between sentences
    distances = calculate_distances(sentences)
    
    # Determine number of cuts needed based on character count
    total_length = sum(len(s) for s in sentences)
    target_cuts = total_length // avg_chunk_size
    
    cut_indices = get_cut_indices(distances, target_cuts)
    
    # Create chunks based on cut points
    chunks = []
    current_chunk = ''
    sentence_pointer = 0
    while sentence_pointer < len(sentences):
        sentence = sentences[sentence_pointer]
        if len(current_chunk) + len(sentence) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ''
            cut_indices = [n+sentence_pointer for n in get_cut_indices(distances[sentence_pointer:], target_cuts-len(chunks))]
            continue
        if len(sentence) < int(min_chunk_size*0.5):
            print(sentence)
        current_chunk += f'\n{sentence}' if current_chunk else sentence
        if sentence_pointer == cut_indices[0]:
            chunks.append(current_chunk.strip())
            current_chunk = ''
            cut_indices.pop(0)
        sentence_pointer += 1
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks



SYSTEM_PROMPT = "You are an assistant specialized in splitting text into thematically consistent sections."

USER_MSG = """The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number.
Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. Try to avoid splitting in the middle of a topic/section/paragraph

{chunked_input}

Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2.
THE CHUNKS MUST BE IN ASCENDING ORDER.
Your response should be in the form: 'split_after: 3, 5'
Respond only with the IDs of the chunks where you believe a split should occur.
YOU MUST RESPOND WITH AT LEAST ONE SPLIT. THESE SPLITS MUST BE IN ASCENDING ORDER AND EQUAL OR LARGER THAN: {current_chunk}
"""

def llm_text_splitter(text,
                      min_chunk_size: int = 800,
                      n_chunks_per_prompt: int = 10,
                      max_retries: int = 5,
                      client: OpenAI | None = None) -> List[str]:
    client = client or OpenAI()
    chunks = recursive_text_splitter(text, min_chunk_size, int(min_chunk_size*0.5), chunk_overlap=0)
    split_indices = []
    current_chunk = 0
    while True:
        if current_chunk >= len(chunks) - 4:
            break
        chunked_input = []
        for i in range(current_chunk, min(len(chunks), current_chunk+n_chunks_per_prompt)):
            chunked_input.append(f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>")
        chunked_input = '\n'.join(chunked_input)
        original_prompt = USER_MSG.format(chunked_input=chunked_input, current_chunk=current_chunk)
        prompt = original_prompt
        final_answer = None
        for _ in range(max_retries):
            result_string = client.chat.completions.create(model='gpt-4o-mini', messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], max_tokens=200, temperature=0.2)
            result_string = result_string.choices[0].message.content
            split_after_line = [line for line in result_string.split('\n') if 'split_after:' in line][0]
            numbers = re.findall(r'\d+', split_after_line)
            numbers = list(map(int, numbers))
            if not (numbers != sorted(numbers) or any(number < current_chunk for number in numbers)):
                final_answer = numbers
                break
            else:
                prompt = original_prompt + f"\nThe previous response of {numbers} was invalid. DO NOT REPEAT THIS ARRAY OF NUMBERS. Please try again." 
        if final_answer is None:
            raise ValueError("Failed to retrieve valid split")
        split_indices.extend(final_answer)
        current_chunk = numbers[-1]
        if len(numbers) == 0:
            break
    chunks_to_split_after = [i - 1 for i in split_indices]
    docs = []
    current_chunk = ''
    for i, chunk in enumerate(chunks):
        current_chunk += chunk + ' '
        if i in chunks_to_split_after:
            docs.append(current_chunk.strip())
            current_chunk = ''
    if current_chunk:
        docs.append(current_chunk.strip())
    return docs