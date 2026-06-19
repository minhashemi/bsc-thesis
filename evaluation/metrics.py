import jiwer

def calculate_metrics(reference: str, hypothesis: str):
    ref = reference.strip()
    hyp = hypothesis.strip()

    wer = jiwer.wer(ref, hyp)
    cer = jiwer.cer(ref, hyp)

    word_measures = jiwer.process_words(ref, hyp)
    lev_distance = word_measures.substitutions + word_measures.deletions + word_measures.insertions

    return {
        "wer": wer,
        "cer": cer,
        "levenshtein": lev_distance
    }
