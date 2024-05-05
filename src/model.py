from abc import abstractmethod
from data_split import DataSplit

class Model:
    def __init__(self):
        self.paradigms = dict()
    
    @abstractmethod
    def train(self, split: DataSplit) -> None:
        raise ValueError('Method train(split) not implemented.')

    @abstractmethod
    def segment(self, form, feats) -> str:
        raise ValueError('Method segment(form, feats) not implemented')

    def __call__(self, form, feats):
        return self.segment(form=form, feats=feats)
    
    def predictions(self, test) -> list:
        preds = list()
        for word, seg, ana in test:
            feats = ana.split('-')[1:]
            pred_seg = self.segment(word, feats=feats)
            preds.append((seg, pred_seg))
        return preds

    def accuracy(self, test, preds=None, return_errors=False):
        n, c = 0, 0
        errors = list()
        for seg, pred_seg in preds if preds else self.predictions(test):
            n += 1
            if seg == pred_seg:
                c += 1
            else:
                errors.append((seg, pred_seg))
                
        acc = c / n if n > 0 else 0.
        if return_errors:
            return acc, errors
        return acc
    
    def precision_recall_f1(self, test, preds=None):
        tp = 0
        fp = 0
        fn = 0
        for seg, pred_seg in preds if preds else self.predictions(test):
            gt_morphemes = set(seg.split('-'))
            pred_morphemes = set(pred_seg.split('-'))
            tp += len(gt_morphemes.intersection(pred_morphemes))
            fp += len(pred_morphemes.difference(gt_morphemes))
            fn += len(gt_morphemes.difference(pred_morphemes))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 / (1 / precision + 1 / recall)
        return precision, recall, f1

    def evaluate(self, split: DataSplit, return_errors: bool=False):
        preds = self.predictions(split.test) # make predictions for this split

        # compute accuracy and erors
        acc, errors = self.accuracy(test=None, preds=preds, return_errors=True)
        # compute precision, recall, f1
        precision, recall, f1 = self.precision_recall_f1(test=None, preds=preds)
        
        if return_errors:
            return precision, recall, f1, acc, errors
        return precision, recall, f1, acc