"""Build a tiny ONNX model + tokenizer with the SAME interface as the exported DistilBERT
(inputs input_ids/attention_mask int64, output logits [batch, 2]). Lets tests and CI exercise
the real OnnxClassifier code without downloading a 67 MB model.
Words in BAD push the 'toxic' logit up."""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

BAD = ["idiot", "stupid", "moron", "trash"]
THREAT = ["kill"]
GOOD = ["thanks", "great", "hello", "day", "nice", "have", "a", "you", "are", "the", "is"]


def build(model_dir, n_labels=2, thresholds=None):
    """n_labels=2: binary model (softmax). n_labels=6: multi-label model (sigmoid per category)."""
    model_dir = Path(model_dir)
    (model_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + GOOD + BAD + THREAT
    tok = Tokenizer(models.WordPiece({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok.save(str(model_dir / "tokenizer" / "tokenizer.json"))

    # logits = sum over tokens of table[token] ; table row = [non-toxic, toxic]
    if n_labels == 2:
        table = np.zeros((len(vocab), 2), dtype=np.float32)
        table[:, 0] = 1.0
        for w in BAD + THREAT:
            table[vocab.index(w)] = [0.0, 8.0]
    else:   # toxic, severe_toxic, obscene, threat, insult, identity_hate
        table = np.full((len(vocab), 6), -1.0, dtype=np.float32)
        for w in BAD:
            table[vocab.index(w), [0, 4]] = 8.0
        for w in THREAT:
            table[vocab.index(w), [0, 3]] = 8.0
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["table", "input_ids"], ["emb"], axis=0),
            helper.make_node("Cast", ["attention_mask"], ["maskf"], to=TensorProto.FLOAT),
            helper.make_node("Unsqueeze", ["maskf", "ax"], ["mask3"]),
            helper.make_node("Mul", ["emb", "mask3"], ["masked"]),
            helper.make_node("ReduceSum", ["masked", "ax1"], ["logits"], keepdims=0),
        ],
        "fake",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
         helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", n_labels])],
        initializer=[helper.make_tensor("table", TensorProto.FLOAT, table.shape, table.flatten()),
                     helper.make_tensor("ax", TensorProto.INT64, [1], [2]),
                     helper.make_tensor("ax1", TensorProto.INT64, [1], [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(model_dir / "model.int8.onnx"))
    if thresholds:
        (model_dir / "thresholds.yaml").write_text(thresholds)
    return model_dir


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1] if len(sys.argv) > 1 else "fake-model"))
