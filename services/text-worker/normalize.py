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
AMBIGUOUS = {"1": "il", "|": "li", "!": "il", "0": "o", "4": "a", "3": "e"}   # 1 is i (1d10t) or l (10ser)

# 3+ single characters separated by one repeated separator: "i d i o t", "s.t.u.p.i.d"
SPACED_RE = re.compile(r"(?<![^\W_])((?:[^\W_]|[@$!0-9])([ .\-_*])(?:(?:[^\W_]|[@$!0-9])\2){1,}(?:[^\W_]|[@$!0-9]))(?![^\W_])")
RUN_RE = re.compile(r"([^\W\d_])\1{2,}")   # letters only: "555" stays
EDGE_PUNCT = "\"'()[]{}<>,.;:?!*-_~`"


def _strip_accents(text):
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _join_spaced(match):
    sep = match.group(2)
    return match.group(1).replace(sep, "")


def segment(text, vocab, max_word=20):
    """Split a run of letters into the fewest vocabulary words: "astupididiot" -> "a stupid idiot".
    Returns None when it cannot be split completely."""
    low = text.lower()
    n = len(low)
    best = [None] * (n + 1)
    best[0] = []
    for i in range(1, n + 1):
        for j in range(max(0, i - max_word), i):
            w = low[j:i]
            if best[j] is not None and (w in vocab and (len(w) > 1 or w in ("a", "i"))):
                cand = best[j] + [text[j:i]]
                if best[i] is None or len(cand) < len(best[i]):
                    best[i] = cand
    return " ".join(best[n]) if best[n] else None


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

    def _join(self, match):
        joined = _join_spaced(match)
        if self.vocab is None or self.known(joined) or not joined.isalpha():
            return joined
        return segment(joined, self.vocab) or joined

    def _leet_candidates(self, word):
        """Every reading of the leet characters (1 -> i or l), at most 16."""
        options = [AMBIGUOUS.get(c, LEET.get(c, c)) if c in LEET else c for c in word]
        n_amb = sum(len(o) > 1 for o in options)
        if n_amb > 4:
            options = [o[0] for o in options]
        return ["".join(p) for p in itertools.product(*options)]

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
            cands = [self._fix_runs(c) if RUN_RE.search(c) else c for c in self._leet_candidates(word)]
            known = [c for c in cands if self.known(c)]
            for cand in (known or cands[:1]):
                if self._accept(word, cand):
                    word = cand
                    break
        return pre + word + post

    def __call__(self, text):
        t = unicodedata.normalize("NFKC", text).translate(INVISIBLE).translate(HOMOGLYPHS)
        t = _strip_accents(t)
        t = SPACED_RE.sub(self._join, t)
        return " ".join(self._fix_word(tok) for tok in t.split())


def vocab_from_tokenizer(tok):
    """Whole-word entries of a WordPiece vocabulary (DistilBERT has ~23k)."""
    return {w for w in tok.get_vocab() if w.isalpha() and not w.startswith("##")}
