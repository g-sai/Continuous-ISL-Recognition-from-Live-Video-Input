import numpy as np

def word_error_rate(reference, hypothesis):
    
    """
    Calculate Word Error Rate (WER).
    
    Args:
    reference (list): List of words in the reference sentence.
    hypothesis (list): List of words in the hypothesized sentence.
    
    Returns:
    float: WER value
    """

    r, h = len(reference), len(hypothesis)
    distance = np.zeros((r+1, h+1), dtype=int)

    for i in range(r+1):
        distance[i][0] = i
    for j in range(h+1):
        distance[0][j] = j

    for i in range(1, r+1):
        for j in range(1, h+1):
            if reference[i-1] == hypothesis[j-1]:
                distance[i][j] = distance[i-1][j-1]
            else:
                substitution = distance[i-1][j-1] + 1
                insertion = distance[i][j-1] + 1
                deletion = distance[i-1][j] + 1
                distance[i][j] = min(substitution, insertion, deletion)

    return distance[r][h] / len(reference)

def calculate_average_wer(references, hypotheses):
    """
    Calculate the average Word Error Rate (WER) for multiple sentences.
    
    Args:
    references (list of lists): List of reference sentences, each a list of words.
    hypotheses (list of lists): List of hypothesized sentences, each a list of words.
    
    Returns:
    float: Average WER value
    """
    total_wer = 0
    num_sentences = len(references)

    for ref, hyp in zip(references, hypotheses):
        total_wer += word_error_rate(ref, hyp)

    average_wer = total_wer / num_sentences if num_sentences > 0 else 0
    return average_wer

references = [
    [],[] #List of sentences(["I","young","healthy"])
]
hypotheses = [
    [],[] #List of sentences(["I","young","healthy"])
]


average_wer = calculate_average_wer(references, hypotheses)
print(f"Average Word Error Rate: {average_wer:.2f}")

