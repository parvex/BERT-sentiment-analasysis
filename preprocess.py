# cleanup text
import re

import nltk as nltk
from enchant.checker import SpellChecker
import spacy
from consts import TOKEN_MAX_LEN

rep = {'\n': ' ', '\\': ' ', '/': '', '-': ' ',
       '"': ' " ', ',': ' , ', '.': ' . ', '!': ' ! ',
       '?': ' ? ', "n't": " not", "'ll": " will", '*': ' * ',
       '(': ' ( ', ')': ' ) ', "s'": "s '"}
rep = dict((re.escape(k), v) for k, v in rep.items())
pattern = re.compile("|".join(rep.keys()))
nltk.download('punkt')
nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')


class Preprocessing:

    def __init__(self) -> None:
        super().__init__()
        self.nlp = spacy.load("en")

    def preprocess_text(self, original_text: str):
        if original_text == '':
            return ""
        original_text = " ".join(str(original_text).split()[:TOKEN_MAX_LEN])
        # remove http links and numbers
        processed_text = re.sub(r'(https?://(www\.)?[^\s]+)|\d+', "", original_text)
        processed_text = pattern.sub(lambda m: rep[re.escape(m.group(0))], processed_text)
        named_entities = Preprocessing.get_named_entities_list(processed_text)
        for name in named_entities:
            processed_text.replace(name, "")

        incorrect_words, suggested_words = self.identify_incorrect_words(processed_text)
        if len(incorrect_words) == 0:
            processed_text = self._predict_words(processed_text, incorrect_words, suggested_words)

        doc = self.nlp(processed_text)
        text = " ".join([token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_ for token in doc])
        return text

    def _predict_words(self, text, incorrect_words, spell_checker_suggestions):
        for i, word in enumerate(incorrect_words):
            if len(spell_checker_suggestions[i]) > 0:
                text = text.replace(word, spell_checker_suggestions[i][0], 1)
        return text

    @staticmethod
    def identify_incorrect_words(text):
        ignorewords = ["!", ",", ".", "\"", "?", '(', ')', '*', '\'']
        spell_checker = SpellChecker("en_US")
        words = text.split()
        # get incorrect words
        incorrect_words = [w for w in words if not spell_checker.check(w) and w not in ignorewords]
        suggested_words = [spell_checker.suggest(word) for word in incorrect_words]
        return incorrect_words, suggested_words

    @staticmethod
    def get_named_entities_list(text: str):
        chunked = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
        continuous_chunk = []
        current_chunk = []
        for i in chunked:
            if type(i) == nltk.Tree:
                current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            if current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue
        return continuous_chunk


if __name__ == "__main__":
    preprocessing = Preprocessing()
    text = """Great knife.  I think it's skirting the lines of legality in my state because you can flip it open, but it's not a "gravity knife" -- it holds itself closed and it has a locking mechanism to hold the blade open, so it seems to fit within the letter of the law.The "shark tooth" tip is a neat design idea that gives it a VERY sharp point, there are serrated edges towards the handle that help out if you're trying to cut through something like twine or wires, it has a belt clip if you're toting it around as a utility knife but is also small enough to easily fit within your pocket.If you're looking for a good utility "buck knife" you might as well get something like this.  The flipping action and one-finger lock disengage means you can easily use it one handed and it seems rugged enough for everyday use in the tool belt.  No more dropping what you're doing so you can pry the blade out or safely close it, just a little wrist and thumb action will do it.Edit:I've had this knife for almost 2 years now and I still like it.  I use it frequently (primarily for cutting open boxes from Amazon... *cough*) and can't get enough of the wrist action on it.  It's so nice to be able to open and close it without having to pry the blade out 2-handed style like other utility knives.  Can't say I can really comment on the whole "knife fighting" aspect other reviewers mention (don't do much knife fighting myself... you know... busy busy...) but as a simple utility knife, it's hard to beat.  Still sharp, too.Edit 2 (Dec 2007) --Still have the knife.  Still like it.  Another reviewer said this knife was double bladed (I assume that means "sharp on both sides") but it's not.  You might think it was from looking at the picture, but mine, which I ordered from this page a few years ago, is only sharp on the bottom, with the exception of the "shark tooth" tip, which is sharp on both sides, but only for a distance of like 1/8" or so (the very sharp tip is great for getting started on those highly annoying, nearly invulnerable hard plastic packages everything comes in these days)."""
    corrected_text = preprocessing.preprocess_text(text)
    print(corrected_text)
