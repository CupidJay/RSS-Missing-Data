# RSS_pytorch
official pytorch implementation of Random Subspace Sampling (RSS) for classification with missing data.

## Abstract
Many real-world datasets suffer from the unavoidable issue of missing values, and therefore classification with missing data has to be carefully handled since inadequate treatment of missing values will cause large errors. In this paper, we propose a random subspace sampling (RSS) method by sampling missing items from the corresponding feature histogram distributions in random subspaces, which is effective and efficient at different levels of missing data. Unlike most established approaches, RSS does not train on fixed imputed datasets. Instead, we design a dynamic training strategy where the filled values change dynamically by resampling during training. Moreover, thanks to the sampling strategy, we design an ensemble testing strategy where we combine the results of multiple runs of a single model, which is more efficient and resource-saving than previous ensemble methods. Finally, we combine these two strategies with the random subspace method, which makes our estimations more robust and accurate. The effectiveness of the proposed method is well validated by experimental study.

Keywords: missing data, random subspace, ensemble learning, deep neural networks

## Getting Started

### Prerequisites
* python 3
* PyTorch (= 1.2)
* torchvision (= 0.4)
* Numpy
* fancyimpute (pip install fancyimpute)
* impyute (pip install impyute)
* missingpy (pip install missingpy)

### Train Examples
#### Our RSS method
- Default setting : We use letter with 40% missing fraction for example
```
CUDA_VISIBLE_DEVICES="0" python main.py \
--config-file configs/letter.yaml \
MODEL.META_ARCHITECTURE NRSHistogramSampleNet \
MODEL.SAMPLE.FIXEDTRAIN False \
DATASETS.MISSING_FRAC 0.4 \
DATASETS.K_FOLD_NUMBER 1 \
MODEL.SAMPLE.ENSEMBLETEST 20
```

- Ablation: what if we do not use dynamic training:
```
CUDA_VISIBLE_DEVICES="0" python main.py \
--config-file configs/letter.yaml \
MODEL.META_ARCHITECTURE NRSHistogramSampleNet \
MODEL.SAMPLE.FIXEDTRAIN True \
DATASETS.MISSING_FRAC 0.4 \
DATASETS.K_FOLD_NUMBER 1 \
MODEL.SAMPLE.ENSEMBLETEST 20
```


- Vary dataset: what if we use yeast with 60% missing fraction:
```
CUDA_VISIBLE_DEVICES="0" python main.py \
--config-file configs/yeast.yaml \
MODEL.META_ARCHITECTURE NRSHistogramSampleNet \
MODEL.SAMPLE.FIXEDTRAIN True \
DATASETS.MISSING_FRAC 0.6 \
DATASETS.K_FOLD_NUMBER 1 \
MODEL.SAMPLE.ENSEMBLETEST 20
```

- More configurations
```
CUDA_VISIBLE_DEVICES="0" python main.py \
--config-file configs/letter.yaml \
MODEL.META_ARCHITECTURE NRSHistogramSampleNet \
MODEL.SAMPLE.FIXEDTRAIN False \
DATASETS.MISSING_FRAC 0.4 \
DATASETS.K_FOLD_NUMBER 1 \
DATASETS.NUM_BINS 20 \  # if you want to change the number of bins in histogram estimation
MODEL.SAMPLE.ENSEMBLETEST 20 \ # you can 
MODEL.N_MUL 50 \ #if you want to use different values for nMul, the same for other hyper-parameters, e.g., nPer
SOLVER.NUM_EPOCHS 50 \ # if you want to specify the total epochs, the same for other settings, e.g., batch-size
OUTPUT_DIR "Results/uci/letter" \ # specify the output directory for log file and model file
```

#### Other comparison methods

- KNN (or other) imputation methods with NRS architecture:  
```
CUDA_VISIBLE_DEVICES="0" python main.py \
--config-file configs/letter.yaml \
MODEL.META_ARCHITECTURE UciNet \
DATASETS.MISSING_FRAC 0.4 \
DATASETS.K_FOLD_NUMBER 1 \
PREPROCESSING.IMPUTER knn \ #choose from [knn, mean, iterative, softimpute, matrix_factorization, em]
```


- MICE imputation with MLP architecture and ensemble imputation:  
```
CUDA_VISIBLE_DEVICES="0" python main.py \
--config-file configs/letter.yaml \
MODEL.META_ARCHITECTURE UciFCNet \
DATASETS.MISSING_FRAC 0.4 \
DATASETS.K_FOLD_NUMBER 1 \
PREPROCESSING.IMPUTER knn \
MODEL.ENSEMBLEIMP 5 \
```

### Evaluation Examples
- Default setting : corresponds to the default setting in Training examples
```
CUDA_VISIBLE_DEVICES="0" python main.py \
--config-file configs/letter.yaml \
MODEL.META_ARCHITECTURE NRSHistogramSampleNet \
MODEL.SAMPLE.FIXEDTRAIN False \
DATASETS.MISSING_FRAC 0.4 \
DATASETS.K_FOLD_NUMBER 1 \
MODEL.SAMPLE.ENSEMBLETEST 20 \ #you can vary the ensemble test H optionally
TRAIN False
```

## Citation
```
@article{RSS,
   title         = {Random Subspace Sampling for Classification with Missing Data},
   author        = {Yun-Hao Cao and Jianxin Wu},
   year          = {2023},
   journal = {Journal of Computer Science and Technology}
```
