import numpy as np

from word import Word
from data_split import DataSplit

class Dataset:
    def __init__(self, path: str) -> None:
        self.path = path
        self.load()

    def __str__(self) -> str:
        return f'Dataset loaded from {self.path}'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __len__(self) -> int:
        return len(self.words)
    
    def load(self):
        self.words = list() # store Word objects
        self.raw_data = list() # store raw forms
        ws = list() # store frequencies
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                root, word, seg, ana, freq = line.strip().split('\t')
                freq = int(freq)

                self.raw_data.append((root, word, seg, ana, freq))

                root = root.upper() # make the root the upper-case version of the lemma
                feats = ana.split('-')[1:] # extract the features

                # create Word object
                word = Word(form=word, root=root, feats=feats)
                self.words.append(word) # store words
                ws.append(freq) # store freq

        assert(len(self.words) == len(self.raw_data))

        self.pop_idxs = list(range(len(self.words)))
        self.p = np.asarray(ws) / sum(ws)

    def build_train(self, seed, size):
        np.random.seed(seed)

        train_idxs = set(np.random.choice(self.pop_idxs, size=size, replace=False, p=self.p))
        train = list(self.words[idx] for idx in sorted(train_idxs))

        test = list() # test data
        test_p = list()
        train_tgts = list() # train tgts (segmentations) for use by supervised models
        train_as_test = list() # training data in test form, for use in kfold cross-validation
        for idx in self.pop_idxs:
            root, word, seg, ana, freq = self.raw_data[idx]
            if idx not in train_idxs:
                test.append((word, seg, ana))
                test_p.append(self.p[idx])
            else:
                train_tgts.append(seg)
                train_as_test.append((word, seg, ana))

        # check that it worked
        assert(len(train) == len(set(train)))
        assert(len(train) == len(train_tgts))
        assert(len(test) == len(self) - len(train))
        assert(len(train) == size)

        return DataSplit(path=self.path,
                         seed=seed, 
                         train=train, 
                         train_tgts=train_tgts, 
                         train_as_test=train_as_test,
                         test=test,
                         test_p=test_p)
