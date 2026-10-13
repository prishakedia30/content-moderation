from normalize import Normalizer

VOCAB = {"you", "are", "an", "idiot", "stupid", "so", "kill", "the", "in", "loser", "hello", "is"}
n = Normalizer(VOCAB)
h = Normalizer()   # no vocabulary: heuristics only


def test_leetspeak():
    assert n("you are an 1d10t") == "you are an idiot"
    assert n("$tupid") == "stupid"
    assert n("l0$3r!") == "loser!"


def test_stretched_letters():
    assert n("you are sooooo stuuuuupid") == "you are so stupid"
    assert n("killllll") == "kill"


def test_spaced_letters():
    assert n("you are an i d i o t") == "you are an idiot"
    assert n("s.t.u.p.i.d") == "stupid"
    assert n("i-d-i-o-t") == "idiot"


def test_homoglyphs_and_invisible():
    assert n("іdіоt") == "idiot"                    # Cyrillic і, о
    assert n("ｉｄｉｏｔ") == "idiot"                # full-width
    assert n("id​iot") == "idiot"              # zero-width space
    assert n("ìdìöt") == "idiot"                    # accents


def test_normal_text_untouched():
    for s in ["hello, the meeting is in 2024", "b4 we go", "covid19 is over", "I am a person",
              "call me at 555-1234", "A B testing"]:
        assert n(s) == " ".join(s.split()), s


def test_heuristic_mode():
    assert h("1d10t") == "idiot"
    assert h("2024") == "2024"


def test_ambiguous_leet_and_merged_spacing():
    v = VOCAB | {"a", "total", "and", "up", "fuck", "shut"}
    m = Normalizer(v)
    assert m("a total 10$er") == "a total loser"            # 1 read as l, not i
    assert m("you are a s t u p i d i d i o t") == "you are a stupid idiot"
    assert m("a total l.o.s.e.r, shut up") == "a total loser, shut up"
