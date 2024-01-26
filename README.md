# ConCat

## Basic Usage

### Environment

```shell
# create virtual environment
conda create --name concat python=3.7
# activate environment
conda activate concat
# install pytorch==1.12.0
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
# install other requirements
pip install -r requirements.txt
```

## Data

### Observation and prediction time settings:
For weibo dataset：
                     
we use 1800 (0.5 hour) or 3600 (1 hour) or 7200 (2 hour) as observation time.
And we set 3600*24 (86400, 1 day) as prediction time.
                     
For aps   dataset：
                     
we use 365x3 (1095, 3 years) or 365x5+1 (1826, 5 years)or 365x7+1 (2556, 7 years) as observation time.
And we set 365x20+5 (7305, 20 years) as prediction time.

The datasets we used in the paper come from:
- [Weibo](https://github.com/CaoQi92/DeepHawkes) (Cao *et al.*, [DeepHawkes: Bridging the Gap between 
Prediction and Understanding of Information Cascades](https://dl.acm.org/doi/10.1145/3132847.3132973), CIKM, 2017). 
- [APS](https://journals.aps.org/datasets) (Released by *American Physical Society*, obtained at Jan 17, 2019).  


### Run the code
```shell
#Run at the root of this repo:
python setup.py build_ext --inplace
# generate information cascades
python gen_cas.py --data=data/weibo/ --observation_time=1800 --prediction_time=86400
# generate cascade graph
python gen_emb.py --data=data/weibo/ --observation_time=1800 --max_seq=100
#train
python train.py --data=weibo --observation_time=1800 --max_seq=100 --max_events=15000

```
More running options are described in the bash file: "train.sh"




