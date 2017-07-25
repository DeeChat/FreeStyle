# FreeStyle: Lyrics Generation via GAN (in progress)

This project is based on the pytorch version of [ARAE](https://github.com/jakezhaojb/ARAE/tree/master/pytorch).

从云协作下载`FreeStyle.rar`，解压并重命名为如下结构：
- train.py
- ...
- data
  + dictionary
  + lyrics
    + 周杰伦
    + 陈奕迅
    + ....

完整运行：
```bash
python train.py --data_path data/lyrics --dict_file data/dictionary --enc_grad_norm "" --no_earlystopping --batch_size 32
```

自动抽取数据子集（以下随机抽取1000首歌）：
```bash
python train.py --data_path data/lyrics --dict_file data/dictionary --enc_grad_norm "" --no_earlystopping --batch_size 32 --subset 1000
```

训练完成后从模型生成文本：
```bash
python generate.py --model_path output/example --data_path data/lyrics --dict_file data/dictionary
```