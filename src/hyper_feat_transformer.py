import argparse
from itertools import product as cartesian_product

from dataset import Dataset
from feat_transformer import FeatTransformer

class HyperFeatTransformer(FeatTransformer):
    '''
    A model that does a hyperparameter sweep of a FeatTransformer.
    '''
    def train(self, split, temp_dir):
        emb_size_opts = [256, 512]
        dropout_opts = [0.1, 0.3]
        batch_size_opts = [32, 128, 256]
        layer_opts = [1, 2]
        head_opts = [4, 8]
        lr_opts = [0.001]

        kfold_split = split.kfold_split()

        opts_to_acc = dict()
        for opts in list(cartesian_product(*[
            emb_size_opts, 
            dropout_opts, 
            batch_size_opts,
            layer_opts,
            head_opts,
            lr_opts
            ])):
            emb_size, dropout, batch_size, layers, heads, lr = opts
            print(f'Training with emb_size: {emb_size} dropout: {dropout} batch_size: {batch_size} layers: {layers} heads: {heads} lr: {lr}')
            
            tr = FeatTransformer()
            tr.train(
                split=kfold_split,
                max_epochs=10000 // len(split.train),
                valid_size=len(kfold_split.test),
                temp_dir=f'{temp_dir}/tune_{len(opts_to_acc)}',
                # hyperparams
                emb_size=emb_size,
                dropout=dropout,
                batch_size=batch_size,
                layers=layers,
                heads=heads,
                lr=lr,
            )
            # compute accuracy
            acc = float(tr.best_checkpoint.split('val_accuracy=')[-1].replace('.ckpt', ''))
            print(f'Resulted in acc: {acc}')
            opts_to_acc[opts] = acc

        # extract best hyperparam options
        best_opts, best_acc = sorted(opts_to_acc.items(), reverse=True, key=lambda it: it[-1])[0]
        emb_size, dropout, batch_size, layers, heads, lr = best_opts
        print(f'Best opts: emb_size: {emb_size} dropout: {dropout} batch_size: {batch_size} layers: {layers} heads: {heads} lr: {lr}')
        print(f'Best acc: {best_acc}')
        # train a full model on the best parameters
        super().train(
            split=split,
            max_epochs=(10 * 10000) // len(split.train),
            valid_size=500,
            temp_dir=temp_dir,
            # hyperparams
            emb_size=emb_size,
            dropout=dropout,
            batch_size=batch_size,
            layers=layers,
            heads=heads,
            lr=lr,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    exp_name_opts = 'hun|mon|fin|tur'

    parser.add_argument('--exp_name', '-e', type=str, required=True, help=exp_name_opts)
    parser.add_argument('--data_path', '-data_path', type=str, required=True, default=None)
    parser.add_argument('--temp_dir', '-temp_dir', type=str, required=True, default=None)
    parser.add_argument('--num_train', '-num_train', type=int, required=False, default=10000)
    parser.add_argument('--seed', '-seed', type=int, required=False, default=0)
    args = parser.parse_args()

    if args.exp_name not in exp_name_opts.split('|'):
        raise ValueError(f'Experiment name must be one of {exp_name_opts}')
    
    print(f'Loading dataset from {args.data_path}')
    ds = Dataset(path=args.data_path)
    print(ds)

    tr = HyperFeatTransformer()

    split = ds.build_train(seed=args.seed, size=args.num_train)

    temp_dir = f'{args.temp_dir}/{args.exp_name}/ts_{args.num_train}/seed_{args.seed}'
    print(f'Output at: {temp_dir}')

    tr.train(split=split, temp_dir=temp_dir)
    precision, recall, f1, acc = tr.evaluate(split)
    print(f'P: {precision} R: {recall} F1: {f1} Acc: {acc}')
