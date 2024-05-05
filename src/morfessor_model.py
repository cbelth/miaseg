import morfessor

from model import Model

class MorfessorModel(Model):
    def train(self, split):
        train = split.train
        io = morfessor.MorfessorIO()
        train = list((1, word.form) for word in train)
        self.mmodel = morfessor.BaselineModel()
        self.mmodel.load_data(train)
        self.mmodel.train_batch()

    def segment(self, form, feats):
        seg, _ = self.mmodel.viterbi_segment(form)
        seg = '-'.join(seg)
        return seg