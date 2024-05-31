# <span style="font-variant:small-caps;">Miaseg</span>

Code for the SCiL 2024 paper *Meaning-Informed Low-Resource Segmentation of Agglutinative Morphology*
```bibtex
@inproceedings{belth2024meaning,
  title={Meaning-Informed Low-Resource Segmentation of Agglutinative Morphology},
  author={Belth, Caleb},
  booktitle={Proceedings of the Society for Computation in Linguistics 2024},
  year={2024}
}
```

## Reproducing Results

The results files from the paper are in `results/`. If you want to reproduce these files or create results for new seeds or training sizes, you can use the script `exp.py`. The script takes four command-line arguments:
```
--exp_name / -e
    - One of hun|mon|fin|tur
--model / -m
    - One of miaseg|morfessor|transformer
--num_train / -num_train (Optional)
    - The number of words to train on
    - By default, the script runs the model once each on 500, 1000, and 10000
--num_seeds / -seeds (Optional)
    - The number of seeds to run on 
    - By default 10
```

For example, to reproduce <span style="font-variant:small-caps;">Miaseg</span>'s results on Turkish, you can run:

```python
python exp.py -e tur -m miaseg
```

Note that each model has a number of required packages to run:
- <span style="font-variant:small-caps;">Miaseg</span> requires [Networkx](https://networkx.org/) (which it uses to topologically sort a graph; see sec. 2.3 of the paper).
- Morfessor requires [Morfessor 2.0](https://morfessor.readthedocs.io/en/latest/)
- Transformer requires [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [yoyodyne](https://github.com/CUNY-CL/yoyodyne).

### Notes on the Transformer comparison model

Running `exp.py` with the model argument as `-m transformer` actually does not run the transformer model. It runs a mock model `TrReader`, which simply reads the predictions of the transformer model and converts them into the evaluation metrics (accuracy, F1, etc.). This is because the transformer takes a long time to run and it is thus desirable to distribute different runs of the model onto different GPUs, save the results of each, and then aggregate them later (as `TrReader` does).

`TrReader` assumes that the result of each transformer run (which is a `preds.txt` file) are in `results/transformer_feats/{exp}/ts_{num_train}/seed_{seed}`, where `{exp}` is one of the language names hun|mon|fin|tur, `{num_train}` is 100, 1000, and 10000, and `{seed}` is 0...10 (all corresponding straight-forwardly to the `exp.py` arguments).

If for some reason you really want to re-run the transformer comparison model, you will need to look at `hyper_feat_transformer.py`, which wraps `feat_transformer` to provide hyperparameter tuning. The arguments are similar but not identical to those of `exp.py`:
```
--exp_name / -e
    - One of hun|mon|fin|tur
--data_path / -data_path
    - A path to the data (e.g., `data/fin/nouns.txt`)
    - This is handy if you, like me, run this on a server where you need to put all the data in a scratch directory with a slurm script
--temp_dir / -temp_dir
    - A path to save the results (should be `results/transformer_feats/{exp}/ts_{num_train}/seed_{seed}`, as described above)
    - This is handy if you, like me, run this on a server where you need to save all the data in a scratch directory before moving it off with a slurm script
--num_train / -num_train (Optional)
    - The number of words to train on
    - By default 10000
--seed / -seed (Optional)
    - The seed to run on 
    - By default 0
```
## Running on Your Own Data

<span style="font-variant:small-caps;">Miaseg</span> is implemented in the Python package [algophon](https://github.com/cbelth/algophon). I recommend that implementation for running the model on your own data. Just `pip install algophon`!
