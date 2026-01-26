1. 下载数据集方式：
```shell
mkdir data
cd data
wget https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML/releases/download/raw_dataset/CWRU_12k.zip
unzip CWRU_12k.zip
cd ..
```

或者
```shell
python download_dataset.py
```

2. 构建知识图谱命令，其中--domain 参数可选[0,1,2,3]
python knowledge_graph.py --domain 1 --data_dir ./data --save_dir ./data/kg --time_steps 1024 --overlap_ratio 0.5 --fs 12000 --smoothing 1e-3

3. 运行 fft 原先方式：
~~~bash
python train_maml.py --dataset CWRU --preprocess FFT --ways 10 --shots 5 --train_domains "0,1,2" --test_domain 3 --iters 10
~~~
日志文件名会带 `_nokg` 后缀（例如 `MAML_CWRU_FFT_10w5s_source012_target3_nokg.log`）。

运行 fft+KG 方式
~~~bash
python train_maml.py --dataset CWRU --preprocess FFT --ways 10 --shots 5 --train_domains "0,1,2" --test_domain 3 --iters 10 --use_kg True --kg_dir ./data/kg
~~~
日志文件名会带 `_kg` 后缀（例如 `MAML_CWRU_FFT_10w5s_source012_target3_kg.log`）。

运行 KG-MLP 方式：
~~~bash
python train_maml.py --dataset CWRU --preprocess PHYS --kg_dir ./data/kg --ways 10 --shots 5 --train_domains "0,1,2" --test_domain 3 --iters 10
~~~

4. DTN 理论下界 baseline（每个 episode 从零初始化并仅用 support 训练），在 load3（domain=3）上评估：

FFT（与当前 CWRU FFT pipeline 一致）：
~~~bash
python train_dtn.py --dataset CWRU --preprocess FFT --test_domain 3 --ways 10 --shots 5 --episodes 100 --dtn_steps 200 --dtn_lr 1e-3
~~~

（可选）PHYS（31维物理特征 pipeline，需要先按第2步生成 KG 文件）：
~~~bash
python train_dtn.py --dataset CWRU --preprocess PHYS --kg_dir ./data/kg --test_domain 3 --ways 10 --shots 5 --episodes 100 --dtn_steps 200 --dtn_lr 1e-3
~~~

