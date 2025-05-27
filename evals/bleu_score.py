from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize

def calculate_bleu_score(references, hypothesis):
   
    """
    Calculate BLEU score for a single sentence.
    
    Args:
    references (list of list): List of reference sentences, each tokenized into a list of words.
    hypothesis (list): Hypothesis sentence tokenized into a list of words.
    
    Returns:
    float: BLEU score
    """

    return sentence_bleu(references, hypothesis)

def calculate_corpus_bleu(all_references, all_hypotheses):
    
    """
    Calculate corpus-level BLEU score for multiple sentences.
    
    Args:
    all_references (list of list of list): List of reference sets, where each reference set is a list of tokenized reference sentences.
    all_hypotheses (list of list): List of hypothesis sentences, each tokenized into a list of words.
    
    Returns:
    float: Corpus-level BLEU score
    """

    return corpus_bleu(all_references, all_hypotheses)


references = [
    [],[] #List of sentences(["I","young","healthy"])
]
hypotheses = [
    [],[] #List of sentences(["I","young","healthy"])
]

tokenized_references = [[word_tokenize(ref)] for ref in references]
tokenized_hypotheses = [word_tokenize(hyp) for hyp in hypotheses]


for i, (ref, hyp) in enumerate(zip(tokenized_references, tokenized_hypotheses)):
    bleu = calculate_bleu_score(ref, hyp)
    print(f"BLEU score for sentence {i+1}: {bleu:.4f}")

corpus_bleu = calculate_corpus_bleu(tokenized_references, tokenized_hypotheses)
print(f"Corpus BLEU score: {corpus_bleu:.4f}")
