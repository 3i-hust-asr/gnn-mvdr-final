### Dataset

###### Some information

- `$data_dir = path/to/dataset`, e.g. `$data_dir=../dataset`
- `$data_dir` contains `simu` and `simu_rirs` directory

###### Generate simulated data

- To prepare (clean, noise, rir) combinations, use command:
```bash
python3 main.py --scenario gen_data --data_dir ../dataset
```

- To mix (clean, noise, rir) to simulated data for train/dev set, use command:
```bash
python3 main.py --scenario mix_wav --data_dir ../dataset
```

###### Generate fixed data for evalution
```bash
python3 main.py --scenario gen_data_eval --data_dir ../dataset
```

### Train

- To run training:

```bash
python3 main.py --scenario train --model mvdr --shuffle --clear_cache --evaluate --batch_size 16 --eval_iter 5
```

- Flag:
	- `--model`: select model (`gnn`, `tencent`, `mvdr`)
	- `--shuffle`: shuffle training data for each epoch
	- `--clear_cache`: delete all previous checkpoints
	- `--evaluate`: activate evaluation process during training
	- `--eval_iter`: evaluate each `eval_iter` epoch


### Evaluate
```python
def evaluate(args):
	model = 'mvdr'
    all_metrics = {}
    for epoch in [6, 5, 4, 3]:
        metric = _evaluate(model, epoch, args)
        all_metrics[f'epoch_{epoch}'] = metric
    with open(f'ckpt/{model}.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)
```

- TODO:
	- change `epoch` variable at `line 72` in file `scenario/evaluate/evaluate.py`.