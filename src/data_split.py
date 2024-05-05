import random

class DataSplit:
    def __init__(self, path: str, seed: int, train: list, train_tgts: list, train_as_test: list, test: list, test_p: list=None):
        self.path = path
        self.seed = seed
        self.train = train
        self.train_tgts = train_tgts
        self.train_as_test = train_as_test
        self.test = test
        self.test_p = test_p

    def __str__(self) -> str:
        return f'DataSpit (seed: {self.seed}, train: {len(self.train)}, test: {len(self.test)})'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def kfold_split(self, train_frac=0.8):
        random.seed(self.seed)
        num_train = round(train_frac * len(self.train))
        
        idxs = list(range(len(self.train)))
        random.shuffle(idxs)
        train_idxs = idxs[:num_train]
        test_idxs = idxs[num_train:]

        # build train
        train = list(self.train[idx] for idx in train_idxs)
        train_tgts = list(self.train_tgts[idx] for idx in train_idxs)
        train_as_test = list(self.train_as_test[idx] for idx in train_idxs)
        # build test
        test = list(self.train_as_test[idx] for idx in test_idxs)

        return DataSplit(path=self.path,
                         seed=self.seed, 
                         train=train, 
                         train_tgts=train_tgts, 
                         train_as_test=train_as_test,
                         test=test)