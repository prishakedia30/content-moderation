import unittest
from spans import Word, find_spans

def W(text, i): return Word(i * 0.5, i * 0.5 + 0.4, text)
def fake_score(text):  # toxic iff 'idiot' is in the text
    return 0.95 if "idiot" in text.lower().split() else 0.05

class T(unittest.TestCase):
    def test_clean_gives_no_spans(self):
        ws = [W(t, i) for i, t in enumerate("have a nice day my friend".split())]
        self.assertEqual(find_spans(ws, fake_score), [])
    def test_window_narrowed_to_toxic_word(self):
        ws = [W(t, i) for i, t in enumerate("you are such an idiot sometimes truly".split())]
        sp = find_spans(ws, fake_score)
        self.assertEqual(len(sp), 1)
        self.assertEqual(sp[0].text, "idiot")
        self.assertAlmostEqual(sp[0].start_s, 4 * 0.5 - 0.05)
    def test_wordlist_catches_what_model_misses(self):
        ws = [W(t, i) for i, t in enumerate("well that is shit really".split())]
        self.assertEqual(find_spans(ws, fake_score, {"shit"}, use_wordlist=False), [])
        sp = find_spans(ws, fake_score, {"shit"})
        self.assertEqual([s.text for s in sp], ["shit"])
    def test_nearby_spans_merge(self):
        ws = [W(t, i) for i, t in enumerate("idiot idiot ok fine".split())]
        self.assertEqual(len(find_spans(ws, fake_score)), 1)
    def test_short_transcript(self):
        ws = [W("idiot", 0)]
        self.assertEqual(len(find_spans(ws, fake_score)), 1)
        self.assertEqual(find_spans([], fake_score), [])

if __name__ == "__main__":
    unittest.main()
