# HGM-GIF

Code for AI Open Paper: [Heterogeneous Graph Knowledge Enhanced Stock Market Prediction](https://www.sciencedirect.com/science/article/pii/S2666651021000243)

Some code are borrowed from [HeterSumGraph](https://github.com/dqwang122/HeterSumGraph). Thanks for their work.



## Dependency

- Python 3.6+
- [PyTorch](https://pytorch.org/) 1.2+
- [DGL](http://dgl.ai) 0.4+



## Data

The data format should be jsonlines, each line should be like this:

```json
{
  "sentences": ["sentence1", "sentence2", ...],
  "events": [["subj_1", "v_1", "obj1"], ["subj_2", "v_2", "obj2"], ...],
  "lable": 1,
  "crop": "apple inc",
  "e2e_edges": [[0, 1], [1, 3], ...],
  "e2s_edges": [[0, 1], [1, 2], ...]
}
```

- sentences: the set of sentences in news documents;
- events: the set of event triples extracted from sentences;
- crop: the corporation whose stock prices need to be predicted;
- label: 0 and 1 represents the stock price of the "crop" will decline and rise, respectively;
- e2e_edges: connections between two events, [0, 1] represents the first event is connected with the second one;
- e2s_edges: connections between sentences and events, [0, 1] represents the first event is connected with the second sentence.

We use [CoreNLP](http://standfordnlp.github.io/CoreNLP/) for event extraction.

TF-IDF files generation can refer to [HeterSumGraph](https://github.com/dqwang122/HeterSumGraph).

Raw Data can refer to [here](https://github.com/sudy/coling2018)
Processed data is stored in the `./data/` directory.

## Train

For training, you can run the commands like this:

```shell
python train_stock.py
```



## Evaluation

For evaluation, you can run commands like this:

```shell
python auc.py
```



## Citation

If you find the paper or the resource is useful, please cite our work in your paper:

- [https://www.sciencedirect.com/science/article/pii/S2666651021000243](https://www.sciencedirect.com/science/article/pii/S2666651021000243)

```tex
@article{xiong2021heterogeneous,
  title={Heterogeneous graph knowledge enhanced stock market prediction},
  author={Xiong, Kai and Ding, Xiao and Du, Li and Liu, Ting and Qin, Bing},
  journal={AI Open},
  volume={2},
  pages={168--174},
  year={2021},
  publisher={Elsevier}
```

