class Word:
    def __init__(self, form, root, feats):
        self.form = form
        self.root = root
        self.feats = tuple(feats)
        
    def __str__(self):
        return f'{self.form} ({self.root} + {self.feats})'
    
    def __repr__(self):
        return self.__str__()
        
    def __eq__(self, other):
        return self.__str__() == other.__str__()
    
    def __neq__(self, other):
        return not self.__eq__(other)
    
    def __lt__(self, other):
        return self.form < other.form
    
    def __hash__(self):
        return hash(self.__str__())
    
    def num_feat_diffs(self, other) -> int:
        '''
        Computes the number of features that differ between :self: and :other:, where a difference is either of:
            (1) a feature that :self: has but :other: does not
            (2) a feature that :other: has but :self: does not
        '''
        if type(other) is not Word:
            raise ValueError(f'{other} is not of type Word')
        union = set(self.feats).union(other.feats)
        intersection = set(self.feats).intersection(other.feats)
        return len(union.difference(intersection))