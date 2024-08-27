import nltk
nltk.download('punkt')
from nltk import sent_tokenize
from typing import List


# wrapper sentence_tokenizer nltk
class NLTKSegmenter():
    
    def predict(self, text: str) -> List[str]:
        return sent_tokenize(text, language='italian')

