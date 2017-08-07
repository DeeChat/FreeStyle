## 使用LDA对生成歌词进行评估

第三方库选择: 
* [GibbsLDA++](http://gibbslda.sourceforge.net/), [github](https://github.com/mrquincle/gibbs-lda)

使用流程:
* 训练LDA模型:
    - Step1. 首先在有道云协作下载`netease-data.json.tar.gz`, 解压得到`netease-data.json`, `train_prepare.py`将`netease-data.json`预处理, 去掉停用词, 得到如下格式文件, 第一行为文档个数M, 接下来M行是M个分好词的文本: 
    ```
    [M]
    [document1]
    [document2]
    ...
    [documentM]

    其中: [documenti] = [wordi1] [wordi2] ... [wordiNi]
    ```

    - Step2. 训练LDA模型
    ```bash
    # train
    $ lda -est [-alpha <double>] [-beta <double>] [-ntopics <int>] [-niters <int>] [-savestep <int>] [-twords <int>] -dfile <string>
    
    ./lda -est -ntopics 20 -alpha 2.5 -beta 0.1 -niters 1000 -savestep 100 -twords 30 -dfile lda_netease_models/netease-lda-train.dat 
    ```
    
    - Step3. 增量训练
    ```bash
    # train
    $ lda -estc -dir <string> -model <string> [-niters <int>] [-savestep <int>] [-twords <int>]
    
    ./lda -estc -dir lda_netease_models -model model-01000 -niters 100 -savestep 100 -twords 20  
    ```
    
* 使用LDA模型对新数据进行inference

    - Step1. `test_prepapre.py`将生成的`answer/*.json`歌词处理成如下形式:
    ```
    [M]
    [document1]
    [document2]
    ...
    [documentM]

    其中: [documenti] = [wordi1] [wordi2] ... [wordiNi]
    ```
    
    - Step1. Inference
    ```bash
    # infer
    ./lda -inf -dir lda_models -model model-01000  -twords 20 -dfile test.dat
    ```
    会得到一个test.dat.theta, This file contains the topic-document distributions, i.e., `$p(topic_t|document_m)$`. Each line is a document and each column is a topic
    
   
    - Step2. 计算Perplexity
    
    Infer之后, 在theta后缀的文件中，每个文档会生成n_topics个值, 表示这个文档属于对应的topic的概率, 计算`$perplexiy = 2^{(-\sum{log(p(t))}/N_{topics})}$`
    ```bash
    python evaluate.py --input_file lda_netease_models/gan_answer.dat.theta
    ```
    最后会得到一个gan_answer.dat.theta.eval的文件,每行对应每个文档的ppl
    **发现个问题, 用GibbsLDA++在迭代过程中, 中间模型存在训练文档的p(topic_t | document_m)包含负数的情况**
