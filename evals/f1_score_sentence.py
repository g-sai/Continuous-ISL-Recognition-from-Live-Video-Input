from sklearn.metrics import precision_recall_fscore_support

def calculate_word_level_f1(references, hypotheses):
   
    """
    Calculate word-level precision, recall, and F1 score.
    
    Args:
    references (list of list): List of reference sentences, each a list of words.
    hypotheses (list of list): List of hypothesized sentences, each a list of words.
    
    Returns:
    tuple: (precision, recall, f1_score)
    """
    
    all_ref_words = set(word for sent in references for word in sent)
    
    y_true = []
    y_pred = []
    
    for ref, hyp in zip(references, hypotheses):
        for word in all_ref_words:
            y_true.append(1 if word in ref else 0)
            y_pred.append(1 if word in hyp else 0)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    return precision, recall, f1

references = [
    [],[] #List of sentences(["I","young","healthy"])
]
hypotheses = [
    [],[] #List of sentences(["I","young","healthy"])
]

precision, recall, f1 = calculate_word_level_f1(references, hypotheses)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


