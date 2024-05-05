import os
import argparse
import numpy as np

import sys
sys.path.append('src/')

from dataset import Dataset
from miaseg import MIASEG
from morfessor_model import MorfessorModel
from tr_reader import TrReader

class Exp:
    def __init__(self, path: str, res_path: str, model_builder, num_train: int, num_seeds: int) -> None:
        '''
        path: a path to the data
        '''
        self.model_builder = model_builder
        self.num_train = num_train
        self.num_seeds = num_seeds

        # make paths relative

        self.path = f'{path}'
        self.res_path = f'{res_path}'

    def load_results(self):
        '''
        Load results
        '''
        self.Ps = list()
        self.Rs = list()
        self.f1s = list()
        self.accs = list()

        for seed in range(self.num_seeds):
            with open(f'{self.res_path}/{seed}_P.txt', 'r') as f:
                P = float(f.readlines()[0].strip())
            self.Ps.append(P)

            with open(f'{self.res_path}/{seed}_R.txt', 'r') as f:
                R = float(f.readlines()[0].strip())
            self.Rs.append(R)

            with open(f'{self.res_path}/{seed}_f1.txt', 'r') as f:
                f1 = float(f.readlines()[0].strip())
            self.f1s.append(f1)

            with open(f'{self.res_path}/{seed}_acc.txt', 'r') as f:
                acc = float(f.readlines()[0].strip())
            self.accs.append(acc)


    def run(self, overwrite=True):
        # build results dir
        if overwrite and os.path.exists(self.res_path):
            os.system(f'rm -r {self.res_path}')
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)
        
        ds = Dataset(path=self.path)
        
        for seed in range(self.num_seeds):
            if not overwrite and os.path.exists(f'{self.res_path}/{seed}_acc.txt'):
                continue
            print(f'Running seed {seed}')
            split = ds.build_train(seed=seed, size=self.num_train)
            # build model
            model = self.model_builder()
            # train model
            model.train(split=split)

            # eval model
            precision, recall, f1, acc, errs = model.evaluate(split, return_errors=True)
            self.write(seed, precision, recall, f1, acc, errs)

        return self
    
    def write(self, seed, precision, recall, f1, acc, errs):
        with open(f'{self.res_path}/{seed}_P.txt', 'w') as f:
            f.write(f'{precision}\n')
        with open(f'{self.res_path}/{seed}_R.txt', 'w') as f:
            f.write(f'{recall}\n')
        with open(f'{self.res_path}/{seed}_f1.txt', 'w') as f:
            f.write(f'{f1}\n')
        with open(f'{self.res_path}/{seed}_acc.txt', 'w') as f:
            f.write(f'{acc}\n')
        with open(f'{self.res_path}/{seed}_errs.txt', 'w') as f:
            f.write('seg\tpred_seg\t\n')
            for seg, pred_seg in errs:
                f.write(f'{seg}\t{pred_seg}\n')
    
    def print_res(self):
        self.load_results()

        for metric, res in zip(
            ['P', 'R', 'f1', 'Acc'],
            [self.Ps, self.Rs, self.f1s, self.accs]
        ):
            m = format(np.mean(res), '0.4f')
            v = format(np.std(res), '0.2f')
            print(f'{metric}: ${m} \pm {v}$')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    exp_name_opts = 'hun|mon|fin|tur'
    model_name_opts = 'miaseg|morfessor|transformer'

    parser.add_argument('--exp-name', '-e', type=str, required=True, help=exp_name_opts)
    parser.add_argument('--model', '-m', type=str, required=True, help=model_name_opts)
    parser.add_argument('--num_train', '-num_train', type=int, required=False, default=None)
    parser.add_argument('--num_seeds', '-seeds', type=int, required=False, default=10)
    args = parser.parse_args()

    if args.exp_name not in exp_name_opts.split('|'):
        raise ValueError(f'Experiment name must be one of {exp_name_opts}')

    if args.model == 'miaseg':
        model_builder = lambda: MIASEG()
    elif args.model == 'morfessor':
        model_builder = lambda: MorfessorModel()
    elif args.model == 'transformer':
        model_builder = lambda: TrReader()
    else:
        raise ValueError(f'Model must be one of {model_name_opts}')
    
    path = f'data/{args.exp_name}/nouns.txt'
        
    if args.num_train:
        res_path = f'results/{args.exp_name}/{args.num_train}/{args.model}'
        model = Exp(
            path=path,
            res_path=res_path,
            model_builder=model_builder,
            num_train=args.num_train,
            num_seeds=args.num_seeds
        ).run()
    else:
        for num_train in [500, 1000, 10000]:
            res_path = f'results/{args.exp_name}/{num_train}/{args.model}'
            model = Exp(
                path=path,
                res_path=res_path,
                model_builder=model_builder,
                num_train=num_train,
                num_seeds=args.num_seeds
            ).run()