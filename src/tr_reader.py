from model import Model
from utils import get_project_root_dir

class TrReader(Model):
    '''
    A pretend model that reads the saved predictions of a Transformer model (which is run on a shared server using slurm).
    '''
    def train(self, split) -> None:
        project_root_dir = get_project_root_dir()
        exp = split.path.split('/')[-2]
        num_train = len(split.train)
        seed = split.seed
        preds_path = f'{project_root_dir}/results/transformer_feats/{exp}/ts_{num_train}/seed_{seed}/preds.txt'

        self.preds = list()
        with open(preds_path, 'r') as f:
            for line in f:
                seg, pred_seg = line.strip().split('\t')
                self.preds.append((seg, pred_seg))

    def predictions(self, test) -> list:
        return self.preds