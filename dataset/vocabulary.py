import json

class T5CopyVocabulary(object):

    def __init__(self, vocab_path, tokenizer, sep=','):
        with open(vocab_path) as out:
            self.d_to_w_group = {}
            self.i_to_w = {}
            self.w_to_i = {}
            self.i_to_cls = {}
            self.id_to_category = {}
            self.word_to_category_id = {}
            for idx, line in enumerate(out):
                items = line.strip().split(sep)
                self.d_to_w_group[idx] = []
                for w in items:
                    w = w.lower()
                    assert len(w) > 0, "empty line %s" % line.strip()
                    fg_index = len(self.i_to_w)
                    self.d_to_w_group[idx].append((w, fg_index))
                    self.i_to_w[fg_index] = w
                    self.w_to_i[w] = fg_index
                    self.i_to_cls[fg_index] = idx
                self.id_to_category[len(self.id_to_category)] = items[0]
                self.word_to_category_id[items[0]] = len(self.word_to_category_id)
            self.detection_size = len(self.id_to_category)

        self.token_fg_w = {}
        for (fg_index, w) in self.i_to_w.items():
            token_word = tokenizer(w, return_tensors="np")['input_ids'][0, :-1].tolist()
            self.token_fg_w[fg_index] = token_word

        self.token_class = {}
        for cls_index, w in self.id_to_category.items():
            token_word = tokenizer(w, return_tensors="np")['input_ids'][0, :-1].tolist()
            self.token_class[cls_index] = token_word

    def get_detection_size(self):
        return self.detection_size

    def get_fg_size(self):
        return len(self.i_to_w)

    def get_category(self):
        return self.id_to_category


