import argparse
import random
import os

import pytorch_lightning as pl

from utils import get_project_root_dir
from model import Model

from yoyodyne import (
    train,
    predict,
    util,
)

class FeatTransformer(Model):
    '''
    A Transformer seq2seq model augmented to use morphological features.
    '''
    def train(self, 
              split,
              max_epochs=10,
              valid_size=1000,
              layers=1, 
              heads=4,
              emb_size=128,
              batch_size=32,
              dropout=0.1,
              lr=0.001,
              temp_dir=None,
        ):

        project_root_dir = get_project_root_dir()
        self.temp_dir = f'{project_root_dir}/data/temp' if not temp_dir else temp_dir
        self.model_dir = f'{self.temp_dir}/models'

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        os.system(f'rm {self.temp_dir}/*') # reset temp directory

        self.valid_size = valid_size

        pl.seed_everything(split.seed)

        # generate train file
        self._write_train(split)
        # generate dev file
        self._write_dev(split)

        self.exp = split.path.split('/')[-2]

        # clear old runs
        os.system(f'rm -r {self.model_dir}/{self.exp}/version_*')

        self.max_eopchs = max_epochs
        self.emb_size = emb_size
        self.heads = heads
        self.dropout = dropout
        self.batch_size = batch_size
        self.layers = layers

        args_list = [
            '--experiment', self.exp,
            '--train', self.train_path,
            '--val', self.dev_path,
            '--model_dir', self.model_dir,
            '--accelerator', 'gpu',
            '--features_sep', ',',
            '--features_col', '3',

            # architecture params
            '--arch', 'transformer',
            '--max_epochs', f'{self.max_eopchs}',
            '--embedding_size', f'{self.emb_size}',
            '--source_attention_heads', f'{self.heads}',
            '--dropout', f'{self.dropout}',
            '--batch_size', f'{self.batch_size}',
            '--encoder_layers', f'{self.layers}',
            '--decoder_layers', f'{self.layers}',
            '--learning_rate', f'{lr}'
        ]

        trainer, model, datamodule, args = self.setup(args_list)
        self.trainer = trainer
        self.model = model

        # train and log the best checkpoint.
        self.best_checkpoint = train.train(trainer, model, datamodule, args.train_from)
        util.log_info(f"Best checkpoint: {self.best_checkpoint}")

    def predictions(self, test, batch_size=32, num_workers=1) -> list:
        # generate test file
        self._write_test(test)

        pred_path = f'{self.temp_dir}/pred.txt'
        args_list = [
            '--experiment', self.exp,
            '--model_dir', self.model_dir,
            '--accelerator', 'gpu',
            '--features_sep', ',',
            '--features_col', '3',
            '--checkpoint', self.best_checkpoint,
            '--predict', self.test_path,
            '--output', pred_path,
            '--arch', 'transformer',
            '--batch_size', f'{batch_size}',
        ]
        parser = argparse.ArgumentParser(description=__doc__)
        predict.add_argparse_args(parser)
        args = args = parser.parse_args(args_list)
        datamodule = predict.get_datamodule_from_argparse_args(args)
        model = predict.get_model_from_argparse_args(args)
        trainer = predict.get_trainer_from_argparse_args(args)
        predict.predict(trainer, model, datamodule, pred_path)

        # read predictions
        preds = list()
        with open(pred_path, 'r') as f:
            for line in f:
                preds.append(line.strip())

        preds = list((gt[1], pred_seg) for gt, pred_seg in zip(test, preds))

        # write the decoded results to a file
        self.results_file = f'{self.temp_dir}/preds.txt'
        with open(self.results_file, 'w') as f:
            for seg, pred_seg in preds:
                if pred_seg.strip() == '':
                    pred_seg = '<unk>'
                f.write(f'{seg}\t{pred_seg}\n')

        return preds

    def setup(self, args_list):
        '''
        Mock up of main() in yoyodyne train.py
        '''

        """Trainer."""
        parser = argparse.ArgumentParser(description=__doc__)
        train.add_argparse_args(parser)
        args = parser.parse_args(args_list)
        util.log_arguments(args)
        trainer = train.get_trainer_from_argparse_args(args)
        datamodule = train.get_datamodule_from_argparse_args(args)
        model = train.get_model_from_argparse_args(args, datamodule)

        return trainer, model, datamodule, args

    def _write_train(self, split):
        self.train_path = f'{self.temp_dir}/train.txt'

        with open(self.train_path, 'w') as f:
            for word, seg in zip(split.train, split.train_tgts):
                src = word.form
                feats = ','.join(sorted(word.feats))
                f.write(f'{src}\t{seg}\t{feats}\n')

    def _write_dev(self, split):
        self.dev_path = f'{self.temp_dir}/dev.txt'

        with open(self.dev_path, 'w') as f:
            dev = list(zip(split.train, split.train_tgts))
            if len(dev) < self.valid_size:
                random.shuffle(dev)
            for word, seg in dev[:self.valid_size]:
                src = word.form
                feats = ';'.join(sorted(word.feats))
                f.write(f'{src}\t{seg}\t{feats}\n')

    def _write_test(self, test):
        self.test_path = f'{self.temp_dir}/test.txt'

        with open(self.test_path, 'w') as f:
            for word, seg, ana in test:
                feats = ','.join(ana.split('-')[1:])
                f.write(f'{word}\t{seg}\t{feats}\n')