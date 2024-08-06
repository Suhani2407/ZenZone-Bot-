from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from rouge import Rouge

def summarize_text(text, language='english', sentences_count=3):
    # Initialize the parser with your text
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    
    # Initialize the summarizer
    stemmer = Stemmer(language)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    
    # Get the summary
    summary = []
    for sentence in summarizer(parser.document, sentences_count):
        summary.append(str(sentence))

    return ' '.join(summary)
def evaluate_summary(model_summary, reference_summary):
    # Initialize ROUGE metric
    rouge = Rouge()

    # Calculate ROUGE scores
    scores = rouge.get_scores(model_summary, reference_summary)

    # Extract ROUGE-1 F1 score (you can choose other ROUGE scores as needed)
    rouge_1_f1 = scores[0]['rouge-1']['f']

    return rouge_1_f1

def main():
    # Example text for summarization
    input_text = """
     "I've been struggling a lot lately. I feel like I'm constantly on edge, and my mind is racing with worries and negative thoughts. It's affecting my sleep, my work, and my relationships. I've tried to cope on my own, but it's becoming overwhelming. I'm not sure where to turn or what to do."
    """
    reference_summary= "The user expresses feeling overwhelmed by constant worry and negative thoughts, which are impacting their sleep, work, and relationships."
    # Summarize the text
    model_summary = summarize_text(input_text)
    # Evaluate the model summary against the reference summary using ROUGE
    rouge_score = evaluate_summary(model_summary, reference_summary)
    
     # Display the ROUGE-1 F1 score as a form of "accuracy"
    print(f"ROUGE-1 F1 Score (Model Accuracy): {rouge_score:.4f}")
    print("Summary:")
    print(model_summary)

if __name__ == "__main__":
    main()
