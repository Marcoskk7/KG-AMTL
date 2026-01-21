1. 下载数据集方式：
```shell
mkdir data
cd data
wget https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML/releases/download/raw_dataset/CWRU_12k.zip
unzip CWRU_12k.zip
cd ..
```

2. 构建知识图谱命令，其中--domain 参数可选[0,1,2,3]
python knowledge_graph.py --domain 1 --data_dir ./data --save_dir ./data/kg --time_steps 1024 --overlap_ratio 0.5 --fs 12000 --smoothing 1e-3

3. 运行 fft 原先方式：
~~~bash
python train_maml.py --dataset CWRU --preprocess FFT --ways 10 --shots 5 --train_domains "0,1,2" --test_domain 3 --iters 10
~~~

运行 fft+KG 方式
~~~bash
python train_maml.py --dataset CWRU --preprocess FFT --ways 10 --shots 5 --train_domains "0,1,2" --test_domain 3 --iters 10 --use_kg True --kg_dir ./data/kg
~~~

运行 KG-MLP 方式：
~~~bash
python train_maml.py --dataset CWRU --preprocess PHYS --kg_dir ./data/kg --ways 10 --shots 5 --train_domains "0,1,2" --test_domain 3 --iters 10
~~~

