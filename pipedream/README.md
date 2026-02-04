
```bash
chmod +x data/download_shakespeare.sh
```

```bash
./data/download_shakespeare.sh
```

```bash
pip install -r requirements.txt
```

## Training
```bash
sbatch run_gpt2.slurm
```

## Inference
```bash
sbatch run_generate.slurm
```