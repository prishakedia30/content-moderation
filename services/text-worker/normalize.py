"""Undo common tricks people use to slip toxic words past a filter:
  leetspeak         1d10t      -> idiot
  stretched letters stuuuupid  -> stupid
  spaced letters    i d i o t  -> idiot      (also i.d.i.o.t, i-d-i-o-t, i_d_i_o_t)
  look-alike chars  іdіоt (Cyrillic), ｉｄｉｏｔ (full-width), ìdìöt (accents)
  invisible chars   id​iot (zero-width space)

When the model's vocabulary is known, a word is only rewritten if the rewrite is a real
vocabulary word and the original is not, so normal text ("2024", "b4", "covid19") is left alone.
"""
import itertools
import re
import unicodedata

INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

# Cyrillic / Greek letters that look like Latin ones (NFKC does not map these).
HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "к": "k", "м": "m", "т": "t", "в": "b", "н": "h", "ԁ": "d", "ɡ": "g", "ⅼ": "l",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "К": "K",
    "М": "M", "Н": "H", "В": "B",
    "α": "a", "ο": "o", "ι": "i", "ν": "v", "κ": "k", "τ": "t", "ρ": "p", "ε": "e", "υ": "u",
    "Ο": "O", "Α": "A", "Ι": "I", "Κ": "K", "Τ": "T", "Ε": "E",
})

LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g",
        "@": "a", "$": "s", "!": "i", "|": "l", "+": "t", "€": "e"}
LEET_RE = re.compile("[" + re.escape("".join(LEET)) + "]")

# 3+ single characters separated by one repeated separator: "i d i o t", "s.t.u.p.i.d"
SPACED_RE = re.compile(r"(?<!\S)((?:[^\W_]|[@$!0-9])([ .\-_*])(?:(?:[^\W_]|[@$!0-9])\2){1,}(?:[^\W_]|[@$!0-9]))(?!\S)")
RUN_RE = re.compile(r"([^\W\d_])\1{2,}")   # letters only: "555" stays
EDGE_PUNCT = "\"'()[]{}<>,.;:?!*-_~`"


def _strip_accents(text):
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _join_spaced(match):
    sep = match.group(2)
    return match.group(1).replace(sep, "")


class Normalizer:
    def __init__(self, vocab=None):
        # vocab: set of lowercase whole words the tokenizer knows (no ## pieces). None = heuristics.
        self.vocab = vocab

    def known(self, word):
        return self.vocab is not None and word.lower() in self.vocab

    def _accept(self, original, candidate):
        if candidate == original or not candidate.isalpha() or len(original) < 3:
            return False
        if self.vocab is None or self.known(candidate):
            return not self.known(original)
        # Not in the vocabulary (e.g. "m0r0n" -> "moron" is split into word pieces): still accept
        # when it clearly looks like letters with digits/symbols swapped in, not a code like "abc123".
        return (len(original) >= 4 and sum(c.isalpha() for c in original) * 2 > len(original)
                and (original[0].isalpha() or original[0] in "@$") and not original[-1].isdigit()
                and not re.search(r"[^A-Za-z]{3,}", original) and not self.known(original))

    def _fix_runs(self, word):
        """stuuuupid -> stupid, killlll -> kill: try each run as 1 or 2 letters."""
        runs = list(RUN_RE.finditer(word))
        if not runs:
            return word
        if self.vocab is None or len(runs) > 4:
            return RUN_RE.sub(r"\1\1", word)
        for lengths in itertools.product((1, 2), repeat=len(runs)):
            out, pos = [], 0
            for m, n in zip(runs, lengths):
                out += [word[pos:m.start()], m.group(1) * n]
                pos = m.end()
            cand = "".join(out) + word[pos:]
            if self.known(cand):
                return cand
        return RUN_RE.sub(r"\1\1", word)

    def _fix_word(self, token):
        # keep surrounding punctuation ("idiot!" stays "idiot!"), fix the core
        core = token.strip(EDGE_PUNCT)
        if not core:
            return token
        start = token.find(core)
        pre, post = token[:start], token[start + len(core):]
        word = core
        if RUN_RE.search(word):
            fixed = self._fix_runs(word)
            if self.vocab is None or self.known(fixed) or not self.known(word):
                word = fixed
        has_alpha = any(c.isalpha() for c in word)
        if has_alpha and LEET_RE.search(word):
            cand = "".join(LEET.get(c, c) for c in word)
            cand = self._fix_runs(cand) if RUN_RE.search(cand) else cand
            if self._accept(word, cand):
                word = cand
        return pre + word + post

    def __call__(self, text):
        t = unicodedata.normalize("NFKC", text).translate(INVISIBLE).translate(HOMOGLYPHS)
        t = _strip_accents(t)
        t = SPACED_RE.sub(_join_spaced, t)
        return " ".join(self._fix_word(tok) for tok in t.split())


def vocab_from_tokenizer(tok):
    """Whole-word entries of a WordPiece vocabulary (DistilBERT has ~23k)."""
    return {w for w in tok.get_vocab() if w.isalpha() and not w.startswith("##")}
