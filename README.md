# diffusion-classifier

```python -m torch.distributed.run --nproc_per_node=2 custom_eval_prob_adaptive_distributed.py --dataset image16_uniform_noise --data_dir /jobtmp/ggaonkar/model-vs-human/datasets/uniform-noise/dnn/session-1/ --split test --n_trials 1 --to_keep 50 10 1 --n_samples 50 100 500 --loss l2 --prompt_path prompts/imagenetgs_prompts.csv --batch_size 128```
