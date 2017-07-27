# FreeStyle: Lyrics Generation via GAN (in progress)

ATTENTION: 预训练AE，然后固定AE再训练GAN的代码请先push到seq_ae_gan这个branch。

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
python train.py --data_path data/lyrics --dict_file data/dictionary --enc_grad_norm "" --no_earlystopping --batch_size 32 --cuda
```

自动抽取数据子集（以下随机抽取1000首歌）：
```bash
python train.py --data_path data/lyrics --dict_file data/dictionary --enc_grad_norm "" --no_earlystopping --batch_size 32 --cuda --subset 1000
```

训练完成后从模型生成文本：
```bash
python generate.py --model_path output/example --data_path data/lyrics --dict_file data/dictionary
```

如果在Windows下运行，由于Windows默认编码为gbk，需要把所有的`open(file)`换成`codecs.open(file, 'r', 'utf-8')`。
