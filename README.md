# FreeStyle: Lyrics Generation via GAN (in progress)

运行实验的方式：
- 从网易云下载处理过的数据：`data.json`（这是运行extract.py的结果；这一步环境比较难配，要编译C什么的所以可以直接拿结果来做下一步）
- 构建词典：`python build_dict.py`
- 提取连续区域：`python chunking.py`，主要的输入参数是词表长度
- 训练autoencoder:
```bash
python train_ae.py --data_file chunks.json --dict_file vocab.txt --outf ae --batch_size 64 --split 0.1 --log_interval 100 --cuda
```
- 训练GAN:
```bash
python train.py --data_file chunks.json --dict_file vocab.txt --ae_model output/ae/autoencoder_model_5.pt --ae_args output/ae/args.json --outf gan --batch_size 64 --log_interval 200 --updates 200000 --cuda
```
- 生成少量歌词：
```bash
# 从单个模型生成
python generate.py --ae_args output/ae/args.json --gan_args output/gan/args.json --vocab_file output/ae/vocab.json --ae_model output/ae/autoencoder_model_5.pt --g_model output/gan/gan_gen_model_11.pt --d_model output/gan/gan_disc_model_11.pt --data_path chunks.json --dict_file vocab.txt --noprint --seed 1111 --ngenerations 50 --outf generated/11.txt
# 从保存的模型批量生成
bash gen_evolution.sh
```
- 生成大量歌词
```bash
# 从训练数据中抽样起始句（或替换为从另外的源抽样起始句）
python sample_quiz.py 
# 读取模型根据给定的起始句生成
python answer_quiz.py --quiz_file quiz.json --ae_model output/ae/autoencoder_model_5.pt --g_model output/gan/gan_gen_model_99999.pt --d_model output/gan/gan_disc_model_99999.pt --outf answer.json
# 或读取一批模型批量生成
bash quiz_evolution.sh
# 计算BLEU score
python engine.py # 预先建立搜索引擎加速BLEU计算
python bleu.py --inputf answer.json --outf bleu.txt
# 计算押韵的分数
python rhyme.py
```

目前项目的文件结构还比较混乱，近期会修复。

---
目前主要需要补充的部分：
- 生成部分的代码：将LM和Beam Search融合进来；连续多句的生成
- autoencoder使用整句的EM和F1（而不是孤立地看每个词）作为衡量标准

其他可以补充的部分：
- 用于展示的简单UI

未来需要扩展的部分：
- 读取多句context的模型构建/整首歌生成的模型
- 和音频的数据接合
