import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
import sys
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='cs', preserve_punctuation=True,  with_stress=True)

def text_to_phonemes(text):
  text = text.strip()
  #print("Text before phonemization: ", text)
  ps = global_phonemizer.phonemize([text])
  if not ps:
      return ""
  ps = word_tokenize(ps[0])
  ps = ' '.join(ps)
  #print("Final text after tokenization: ", ps)
  return ps


for line in sys.stdin:
    print(text_to_phonemes(line))
