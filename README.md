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
--exp-name / -e
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


## Running on Your Own Data

<span style="font-variant:small-caps;">Miaseg</span> will soon be implemented in the Python package [algophon](https://github.com/cbelth/algophon). This will make it easy to run the model on your own data. 

Until then, please see `dataset.py` for how to load data and create train/test splits and see `miaseg.py` for how to run the model and use it to segment. For example data formats, see the files in `data/`.
