# cleanup text
import re
from difflib import SequenceMatcher

import nltk as nltk
import torch
from enchant.checker import SpellChecker
from transformers import BertForMaskedLM, BertTokenizer

from consts import PRE_TRAINED_MODEL_NAME, TOKEN_MAX_LEN

rep = {'\n': ' ', '\\': ' ', '-': ' ',
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

    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = BertForMaskedLM.from_pretrained("bert-base-cased")
        # self.model.to(device)

    def preprocess_text(self, original_text: str):
        original_text = " ".join(original_text.split()[:TOKEN_MAX_LEN])
        processed_text = pattern.sub(lambda m: rep[re.escape(m.group(0))], original_text)

        incorrect_words_indices, suggested_words = self.identify_incorrect_words(processed_text)
        if len(incorrect_words_indices) == 0:
            return original_text

        incorrect_words = [processed_text.split()[index] for index in incorrect_words_indices]
        masked_original_text = self.mask_incorrect_words(original_text, incorrect_words)
        processed_text = self.mask_incorrect_words(processed_text, incorrect_words)

        tokenized_text = self.tokenizer.tokenize(processed_text)[:TOKEN_MAX_LEN]

        # For BERT to get the information about sentences
        segments_tensors = self.prepare_segments(tokenized_text)
        tokens_tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)])
        print(len(tokens_tensor[0]))

        # Predict all tokens
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)

        corrected_processed_text = self._predict_words(masked_original_text, predictions, incorrect_words_indices,
                                                       suggested_words)
        return corrected_processed_text

    def _predict_words(self, text, bert_predictions, mask_indices, spell_checker_suggestions):
        for i in range(len(mask_indices)):
            if mask_indices[i] >= TOKEN_MAX_LEN:
                return text
            preds = torch.topk(bert_predictions[0][0][mask_indices[i]], k=50)
            indices = preds.indices.tolist()
            list1 = self.tokenizer.convert_ids_to_tokens(indices)
            list2 = spell_checker_suggestions[i]
            max = 0
            predicted_token = ''
            for word1 in list1:
                for word2 in list2:
                    s = SequenceMatcher(None, word1, word2).ratio()
                    if s is not None and s > max:
                        max = s
                        predicted_token = word1
            text = text.replace('[MASK]', predicted_token, 1)
        return text

    @staticmethod
    def identify_incorrect_words(text):
        personslist = Preprocessing.get_persons_list(text)
        ignorewords = personslist + ["!", ",", ".", "\"", "?", '(', ')', '*', '\'']
        spell_checker = SpellChecker("en_US")
        words = text.split()
        # get incorrect words
        incorrect_words_indices = [i for i, w in enumerate(words) if
                                   not spell_checker.check(w) and w not in ignorewords]
        suggested_words = [spell_checker.suggest(words[index]) for index in incorrect_words_indices]
        return incorrect_words_indices, suggested_words

    @staticmethod
    def get_persons_list(text):
        personslist = []
        for sent in nltk.sent_tokenize(text):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'PERSON':
                    personslist.insert(0, (chunk.leaves()[0][0]))
        return list(set(personslist))

    @staticmethod
    def mask_incorrect_words(text, incorrect_words):
        for w in incorrect_words:
            text = text.replace(w, '[MASK]')
        return text

    @staticmethod
    def prepare_segments(tokenized_text):
        segs = [i for i, e in enumerate(tokenized_text) if e == "."]
        segments_ids = []
        prev = -1
        for k, s in enumerate(segs):
            segments_ids = segments_ids + [k] * (s - prev)
            prev = s
        segments_ids = segments_ids + [len(segs)] * (len(tokenized_text) - len(segments_ids))
        return torch.tensor([segments_ids])


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    preprocessing = Preprocessing(tokenizer)
    text = """Great knife.  I think it's skirting the lines of legality in my state because you can flip it open, but it's not a "gravity knife" -- it holds itself closed and it has a locking mechanism to hold the blade open, so it seems to fit within the letter of the law.The "shark tooth" tip is a neat design idea that gives it a VERY sharp point, there are serrated edges towards the handle that help out if you're trying to cut through something like twine or wires, it has a belt clip if you're toting it around as a utility knife but is also small enough to easily fit within your pocket.If you're looking for a good utility "buck knife" you might as well get something like this.  The flipping action and one-finger lock disengage means you can easily use it one handed and it seems rugged enough for everyday use in the tool belt.  No more dropping what you're doing so you can pry the blade out or safely close it, just a little wrist and thumb action will do it.Edit:I've had this knife for almost 2 years now and I still like it.  I use it frequently (primarily for cutting open boxes from Amazon... *cough*) and can't get enough of the wrist action on it.  It's so nice to be able to open and close it without having to pry the blade out 2-handed style like other utility knives.  Can't say I can really comment on the whole "knife fighting" aspect other reviewers mention (don't do much knife fighting myself... you know... busy busy...) but as a simple utility knife, it's hard to beat.  Still sharp, too.Edit 2 (Dec 2007) --Still have the knife.  Still like it.  Another reviewer said this knife was double bladed (I assume that means "sharp on both sides") but it's not.  You might think it was from looking at the picture, but mine, which I ordered from this page a few years ago, is only sharp on the bottom, with the exception of the "shark tooth" tip, which is sharp on both sides, but only for a distance of like 1/8" or so (the very sharp tip is great for getting started on those highly annoying, nearly invulnerable hard plastic packages everything comes in these days)."""
    corrected_text = preprocessing.preprocess_text(text)
    print(corrected_text)
