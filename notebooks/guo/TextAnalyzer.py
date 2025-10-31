import pandas as pd
import spacy, re

class TextAnalyzer:
    def __init__(self):
        # 加载spacy模型
        self.nlp = spacy.load("en_core_web_md")
        
    def extract_raw_tokens(self, 
        text: str, 
        remove_nouns: bool = False,
        remove_proper_nouns: bool = False,
        remove_adjectives: bool = False,
        remove_adverbs: bool = False,
        remove_verbs: bool = False
    ) -> list[str]:
        if pd.isna(text):
            return []
            
        doc = self.nlp(text)
        meaningful_tokens = []
        
        for token in doc:
            if (
                token.is_punct or 
                token.is_space or 
                token.is_stop or
                len(token.text) < 2
            ):
                continue
                
            if remove_nouns and token.pos_ == "NOUN":
                continue
            if remove_proper_nouns and token.pos_ == "PROPN":
                continue
            if remove_adjectives and token.pos_ == "ADJ":
                continue
            if remove_adverbs and token.pos_ == "ADV":
                continue
            if remove_verbs and token.pos_ == "VERB":
                continue
                
            if token.pos_ in ["NOUN", "PROPN", "ADJ", "ADV", "VERB"]:
                meaningful_tokens.append(token.lemma_.lower())
                
        return meaningful_tokens

    def extract_noun_chunks(self, 
        text: str, 
    ) -> list[str]:
        if pd.isna(text):
            return []
        doc = self.nlp(text)
        noun_chunks = []

        for chunk in doc.noun_chunks:
            # remove leading determiners, e.g. "the", "a", "an", "this"
            start = chunk.start
            end = chunk.end
            while start < end and doc[start].pos_ == "DET":
                start += 1
            
            clean_text = doc[start:end].text.lower().strip()

            parts = re.split(r"[:–—|]", clean_text)
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                if re.search(r"^[\W_]+$", part):
                    continue

                noun_chunks.append(part)
                
        return noun_chunks

def create_sample_data():
    sample_texts = [
        "Machine learning models require extensive training data and computational resources for optimal performance.",
        "Deep learning algorithms, particularly convolutional neural networks, have revolutionized computer vision applications.",
        "Natural language processing techniques leverage machine learning to understand and generate human language.",
        "Data scientists employ statistical methods and machine learning algorithms to extract insights from complex datasets.",
        "The deployment of AI systems in production environments requires robust monitoring and maintenance protocols.",
        "Transfer learning allows pre-trained models to be adapted for specific tasks with limited additional training.",
        "Explainable AI methods help interpret the decisions made by complex machine learning models.",
        "Reinforcement learning algorithms learn optimal strategies through interaction with environments and reward signals."
    ]
    return pd.DataFrame({'content': sample_texts})

if __name__ == "__main__":
    analyzer = TextAnalyzer()
    sample_data = create_sample_data()
    print(analyzer.extract_raw_tokens(sample_data['content'][0]))