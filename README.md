# FreeStyle: Lyrics Generation via GAN (in progress)

This project is based on the pytorch version of [ARAE](https://github.com/jakezhaojb/ARAE/tree/master/pytorch).

完整运行：
```bash
python train.py --data_path data/lyrics --dict_file data/dictionary --enc_grad_norm "" --no_earlystopping
```

调试模式：
```bash
python train.py --data_path data/lyrics --dict_file data/dictionary --enc_grad_norm "" --no_earlystopping --batch_size 2 --subset 100
```

训练完生成文本：
```bash
python generate.py --model_path output/example --data_path data/lyrics --dict_file data/dictionary
```