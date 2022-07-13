## DATASET
Download the original dataset from [ml-1m](http://files.grouplens.org/datasets/movielens/) and [Amazon](http://jmcauley.ucsd.edu/data/amazon/index_2014.html). Then follow the steps below to process the dataset.

### Process:
- Divide the original dataset into three non-overlapping parts: shadow_dataset, target dataset and dataset for vectorization, which are used for training shadow recommender system, target recommender system and obtaining the item representation respectively. 
```bash
cd DL-MIA/DL-MIA-SR/Dataset/
python process_amazon_Beauty.py
python process_ml1m.py
```
- Get the latent representation of items by LFM(Latent factor model).
 ```bash
cd RecSys-master/
python mainCF.py
```
- Train shadow model and target model using shadow dataset and target dataset respectively. Then generate recommendaiton lists for member and nonmember(Based on popularity).
 ```bash
# e.g. BERT4Rec
cd ../Recommender/BERT4Rec-Pytorch-master/
sh BERT4Rec.sh
```
Note that for different recommendation models, further dataset processing and file path changes are required here. We provide the corresponding code, which is not described in detail here.
- Generate difference feature vector.
 ```bash
# e.g. BERT4Rec
cd ../../Biased-Attack/SMDD/AttackModel/
sh attack.sh
```
- Process data to match model inputs.
 ```bash
# e.g. BERT4Rec
cd ../../../Joint-Training/
sh dataPreProcess.sh
```