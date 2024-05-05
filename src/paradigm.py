class Paradigm:
    def __init__(self, root):
        self.root = root
        self.words = set()
        
    def __str__(self):
        return self.root
    
    def __repr__(self):
        return self.__str__()
    
    def __len__(self):
        return len(self.words)
    
    def add_word(self, word_form):
        self.words.add(word_form)

    def get_unmarked(self):
        return list(word for word in self.words if len(word.feats) == 0)
        
    def get_one_diff_pairs(self):
        pairs = list()
        words = sorted(self.words)
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w1 = words[i]
                w2 = words[j]
                if w1.num_feat_diffs(w2) == 1:
                    if len(w1.feats) < len(w2.feats):
                        pairs.append((w1, w2))
                    else:
                        pairs.append((w2, w1))
        return pairs