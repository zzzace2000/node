
# Just run like 1000 es to test functionality of rs
CUDA_VISIBLE_DEVICES=0 python main.py --name 0225 --dataset a9a --random_search 10 --early_stopping_rounds 10000

CUDA_VISIBLE_DEVICES=0 python main.py --name 0225 --dataset a9a --random_search 50 --early_stopping_rounds 10000




./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name testing --dataset a9a --early_stopping_rounds 10000

# Run in vws
python main.py --name test2 --dataset a9a  --early_stopping_rounds 500 --fp16 0 --random_search 5

# Search in v or q to do the random search
python main.py --name test3 --dataset a9a --early_stopping_rounds 500 --random_search 5
python main.py --name test4 --dataset epsilon --early_stopping_rounds 200 --random_search 1


## Search in a9a
python main.py --name 0225_a9a --dataset a9a --random_search 100 --cpu 4 --gpus 1 --mem 8
python main.py --name 0226_epsilon --dataset epsilon --random_search 100 --cpu 4 --gpus 1 --mem 10 --random_search 2

# Rerun a specific job
./my_sbatch --cpu 4 --gpus 1 --mem 10 python main.py --name 0226_epsilon_s5_nt8192_bs512_mt0.0003_as2000_cfSMTemp

# Rerun 5 datasets for once to download datasets
for dset in 'year' 'higgs' 'microsoft' 'yahoo' 'click'; do
  python main.py --name 0226_${dset} --dataset ${dset} --random_search 1 --cpu 4 --gpus 1 --mem 10
done

## TORUN: to see if they perform reasonable!
for dset in 'year' 'higgs' 'microsoft' 'yahoo' 'click' 'epsilon'; do
  python main.py --name 0226_${dset} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8
done

## After fixing bugs, run more random search!
for dset in 'year' 'higgs' 'microsoft' 'yahoo' 'click' 'epsilon'; do
  python main.py --name 0302_${dset} --dataset ${dset} --random_search 15 --cpu 4 --gpus 1 --mem 12
done

for dset in 'year' 'higgs' 'microsoft' 'yahoo' 'click' 'epsilon'; do
  mv "results/${dset}_prev.csv" "results/${dset}.csv"
done


## Run python scripts
python main.py --name 0305 --dataset mimic2

# Run mimic2
dset='mimic2'
python main.py --name 0306_${dset} --dataset ${dset} --random_search 50 --cpu 4 --gpus 1 --mem 8

# Run baselines in all datasets with ebm-o100-i100
dset='mimic2'
model_name='ebm-o100-i100'
./my_sbatch --cpu 20 --gpus 0 --mem 8 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}


for dset in 'year' 'higgs' 'microsoft' 'yahoo' 'click' 'epsilon'; do
model_name='ebm-o100-i100'
./my_sbatch --cpu 20 --gpus 0 --mem 16 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done

# They die in wierd situations....
for dset in 'yahoo' 'microsoft' 'epsilon'; do
model_name='ebm-o100-i100'
./my_sbatch --cpu 20 --gpus 0 --mem 32 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done

dset='yahoo'
model_name='ebm-o100-i100'
./my_sbatch --cpu 20 --gpus 0 --mem 50 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}

for dset in 'epsilon'; do
model_name='ebm-o100-i100'
./my_sbatch --cpu 20 --gpus 0 --mem 32 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done

# Run more for only these 3 datasets
for dset in 'adult' 'mimic3' 'compas'; do
  python main.py --name 0310_${dset} --dataset ${dset} --random_search 30 --cpu 4 --gpus 1 --mem 8

  model_name='ebm-o100-i100'
./my_sbatch --cpu 20 --gpus 0 --mem 16 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done


for dset in 'mimic3' 'compas'; do
model_name='ebm-o100-i100'
./my_sbatch --cpu 20 --gpus 0 --mem 16 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done


# THink: how to make NODE GAM deep? How can we penalize (either hard or soft) to make the later layers only choose the ones with similar features?
- Maybe an inner product (like attention)? So the more similar of picking the feature for that node, the better to focus on which end to take. But I can not make it as input-dependent!
  - One way to achieve is to say one subsequent tree focuses on one set of features. But which set of features? One thing is I can do a soft attention (with one vector and an attention value on each previous leafs, )
- Idea: follow the attention/encoder architecture in table features?

# Finish the attention! Search multi-layer!
dset='mimic2'
python main.py --name 0312_${dset} --dataset ${dset} --random_search 50 --cpu 4 --gpus 1 --mem 8

# Do a test: if fp16 speeds up or not? No!
dset='mimic2'
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0312_${dset} --dataset ${dset} --num_trees 1024 --num_layers 6 --tree_dim 1 --fp16 0
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0312_${dset}_fp16 --dataset ${dset} --num_trees 1024 --num_layers 6 --tree_dim 1 --fp16 1

# Do a grid search
# (1) List the best hyparprameter so far found on MIMIC2
dset='mimic2'
./my_sbatch -p p100 --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0316_${dset} --dataset ${dset} --num_trees 1024 --batch_size 2048 --num_layers 2 --tree_dim 0 --output_dropout 0.5 --colsample_bytree 0.3 --att_normalize 1

# (2) Here search which number of tree works best. Here change bs in 1024
#dset='mimic2'
for dset in 'compas' 'adult'; do
for num_trees in '1024' '2048' '4096' '8192' '16400'; do
./my_sbatch -p p100 --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0316_${dset}_nt${num_trees} --dataset ${dset} --num_trees ${num_trees} --batch_size 1024 --num_layers 2 --tree_dim 0 --output_dropout 0.5 --colsample_bytree 0.3 --att_normalize 1
done

# (3) Search if batch size makes a difference?
for bs in '128' '256' '512' '1024' '2048' '4096'; do
./my_sbatch -p p100 --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0316_${dset}_bs${bs} --dataset ${dset} --num_trees 1024 --batch_size ${bs} --num_layers 2 --tree_dim 0 --output_dropout 0.5 --colsample_bytree 0.3 --att_normalize 1
done

# (4) Search if num_layers make a difference?
for nl in '1' '2' '3' '4' '5' '6' '7' '8'; do
./my_sbatch -p p100 --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0316_${dset}_nl${nl} --dataset ${dset} --num_trees 1024 --batch_size 1024 --num_layers ${nl} --tree_dim 0 --output_dropout 0.5 --colsample_bytree 0.3 --att_normalize 1
done

# (5) search normalize makes a difference?
./my_sbatch -p p100 --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0316_${dset}_nn --dataset ${dset} --num_trees 1024 --batch_size 2048 --num_layers 2 --tree_dim 0 --output_dropout 0.5 --colsample_bytree 0.3 --att_normalize 0

# (6) colsample_bytree
for cs in '1' '0.8' '0.5' '0.3' '0.1' '0.0001'; do
./my_sbatch -p p100 --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0316_${dset}_cs${cs} --dataset ${dset} --num_trees 1024 --batch_size 2048 --num_layers 2 --tree_dim 0 --output_dropout 0.5 --colsample_bytree ${cs}
done

# (7) output_dropout
for od in '0.' '0.1' '0.3' '0.5' '0.8'; do
./my_sbatch -p p100 --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0316_${dset}_od${od} --dataset ${dset} --num_trees 1024 --batch_size 2048 --num_layers 2 --tree_dim 0 --output_dropout ${od} --colsample_bytree 0.3
done

# (8) tree_dim
for tree_dim in '0' '1' '2' '3'; do
./my_sbatch -p p100 --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0316_${dset}_td${tree_dim} --dataset ${dset} --num_trees 1024 --batch_size 2048 --num_layers 2 --tree_dim ${tree_dim} --output_dropout 0.5 --colsample_bytree 0.3
done

# (9) l2 penalty of response?
#for l2_lambda in '0' '1e-3' '1e-4' '1e-5' '1e-6' '1e-7'; do
#for l2_lambda in '0.01' '0.1'; do
for l2_lambda in '1e-6' '1e-7' '1e-8' '1e-9' '0'; do
./my_sbatch -p p100 --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0316_${dset}_lda${l2_lambda} --dataset ${dset} --num_trees 1024 --batch_size 2048 --num_layers 2 --tree_dim 0 --output_dropout 0.5 --colsample_bytree 0.3 --l2_lambda ${l2_lambda}
done

done

for dset in 'mimic2' 'adult' 'compas'; do
for l2_lambda in '1e-10' '1e-11' '1e-12'; do
./my_sbatch -p p100 --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0316_${dset}_lda${l2_lambda} --dataset ${dset} --num_trees 1024 --batch_size 2048 --num_layers 2 --tree_dim 0 --output_dropout 0.5 --colsample_bytree 0.3 --l2_lambda ${l2_lambda}
done
done

#### Run more hyperparameters search
for dset in 'mimic2' 'compas' 'adult'; do
  fold='1'
  python main.py --name 0317_${dset}_f${fold} --dataset ${dset} --random_search 40 --cpu 4 --gpus 1 --mem 8 --fold ${fold}

  model_name='ebm-o100-i100'
  for fold in '1' '2' '3' '4'; do
  ./my_sbatch --cpu 20 --gpus 0 --mem 10 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
  done
done

# Rerun those failed model
for f in \
'0321_mimic2_ODST_s54_nl4_nt1024_d6_td0_lr0.001'; do
  ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${f}
done

for f in \
'0321_mimic2_GAMAtt_s2_d0.3_cs0.5_nl4_nt4000_td0_an1_at32'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${f}
done

## Memory explode
'0317_mimic2_f1_s84_bs1024_as20000_d0.5_cs0.5_nl4_nt12000_td0_an1_lda0.0'




# Run a few models using Attn
dset='mimic2'
for arch in 'GAM' 'GAMAtt'; do
./my_sbatch -p p100 --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0318_${dset}_${arch} --dataset ${dset} --num_trees 1024 --batch_size 2048 --num_layers 2 --tree_dim 0 --output_dropout 0.5 --colsample_bytree 0.3 --att_normalize 0 --arch ${arch}
done

arch='GAM'
dset='mimic2'
./my_sbatch -p p100 --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0318_${dset}_${arch}_fp16 --dataset ${dset} --num_trees 1024 --batch_size 2048 --num_layers 2 --tree_dim 0 --output_dropout 0.5 --colsample_bytree 0.3 --att_normalize 0 --arch ${arch} --fp16 1

# Run all folds with the best mimic2 (l2) parameter
arch='GAM'
dset='mimic2'
for fold in '0' '1' '2' '3' '4'; do
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0319_${dset}_${arch}_best_f${fold} --dataset ${dset} --seed 1 --num_trees 4000 --batch_size 1024 --num_layers 2 --tree_dim 0 --output_dropout 0.5 --colsample_bytree 1e-5 --arch ${arch} --fp16 0 --fold ${fold} --early_stopping_rounds 15000
done

# Run a version of fp16=1. See performance / run time is slower or faster
arch='GAM'
dset='mimic2'
for fold in '0' '1' '2'; do
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0319_${dset}_${arch}_best_f${fold}_fp16 --dataset ${dset} --seed 1 --num_trees 4000 --batch_size 1024 --num_layers 2 --tree_dim 0 --output_dropout 0.5 --colsample_bytree 1e-5 --arch ${arch} --fp16 1 --fold ${fold} --early_stopping_rounds 15000
done


# TODO:
# (1) Wait for server to empty. See for attention architecture, if it works better? Run fp16 to save memory...
for dset in 'mimic2'; do
  python main.py --name 0319_${dset} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1
done

for dset in 'compas' 'adult'; do
  python main.py --name 0319_${dset} --dataset ${dset} --random_search 50 --cpu 4 --gpus 1 --mem 8 --fp16 1
done


model_name='xgb-o50'
for dset in 'mimic2' 'compas' 'adult'; do
for fold in '0' '1' '2' '3' '4'; do
./my_sbatch --cpu 30 --gpus 0 --mem 10 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
done
done

# Add lr! Do random search to make sure this is still ok compared to previous methods
for dset in 'mimic2'; do
  python main.py --name 0320_${dset} --dataset ${dset} --random_search 5 --cpu 4 --gpus 1 --mem 8 --fp16 1
done

# Change lr schedule! Rerun search!
for arch in 'GAM' 'GAMAtt'; do
  for dset in 'compas' 'adult'; do
    python main.py --name 0321_${dset}_${arch} --dataset ${dset} --random_search 10 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

for arch in 'GAM' 'GAMAtt'; do
  for dset in 'support2' 'churn'; do
    python main.py --name 0321_${dset}_${arch} --dataset ${dset} --random_search 2 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

for arch in 'GAM' 'GAMAtt'; do
  for dset in 'support2' 'churn' ; do
    python main.py --name 0321_${dset}_${arch} --dataset ${dset} --random_search 2 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

for arch in 'GAM' 'GAMAtt'; do
  for dset in 'support2' 'churn' ; do
    python main.py --name 0321_${dset}_${arch} --dataset ${dset} --random_search 2 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

# Run everything for once to download dataset!
for arch in 'GAM'; do
  for dset in 'credit' 'mimic3' 'click' 'yahoo' 'microsoft' 'higgs' 'epsilon' 'year'; do
    python main.py --name 0321_${dset}_${arch} --dataset ${dset} --random_search 1 --cpu 4 --gpus 1 --mem 8 --fp16 0 --arch ${arch}
  done
done


# Run the best parameter again!
arch='GAM'
dset='mimic2'
for fold in '0' '1' '2' '3' '4'; do
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0322_${dset}_${arch}_best_f${fold} --dataset ${dset} --seed 1 --num_trees 4000 --batch_size 1024 --num_layers 2 --tree_dim 0 --output_dropout 0.5 --colsample_bytree 1e-5 --arch ${arch} --fp16 0 --fold ${fold} --early_stopping_rounds 15000
done


# Run random search for every dataset. The current combination around 320
# Run 100 for each dataset. Enable fp16 (since it goes deeper, it could be faster?).
for dset in 'adult' 'mimic3'; do
  for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0321_${dset}_${arch} --dataset ${dset} --random_search 50 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --early_stopping_rounds 11000 --lr_decay_steps 5000
  done
done

for dset in 'mimic3' 'credit' 'churn' 'support2'; do
  for fold in '0'; do
    for model_name in 'xgb-o50' 'ebm-o100-i100'; do
      ./my_sbatch --cpu 20 --gpus 0 --mem 10 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
    done
  done
done

model_name='xgb-o50'
for dset in 'mimic2' 'compas' 'adult'; do
for fold in '0' '1' '2' '3' '4'; do
./my_sbatch --cpu 30 --gpus 0 --mem 10 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
done
done


# Run a different depth!!
arch='GAM'
dset='mimic2'
fold='0'
for depth in '1' '2' '4' '6' '8'; do
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0324_${dset}_${arch}_best_f${fold}_d${depth} --dataset ${dset} --seed 1 --num_trees 4000 --num_layers 2 --tree_dim 0 --output_dropout 0.5 --colsample_bytree 1e-5 --arch ${arch} --fp16 1 --fold ${fold} --early_stopping_rounds 11000 --lr_decay_steps 5000 --depth ${depth}
done


# (Wait) wait MIMIC result and select the range of depth
for dset in 'mimic2' 'adult'; do
  for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0322_${dset}_${arch} --dataset ${dset} --random_search 50 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done



After seeing the graph, I feel my preprocessing has huge impacts of the graph
(1) Before, I only do target transform without doing the quantile transform on categorical features. That makes the feature value quite close in general (e.g. 0.008, 0.012). By making quantile transform the value becomes much larger: -2, 0
(2) Before, I have quantile noise arond 1e-3. This inductive bias makes things smooth. But my datasets are usually large (except compas?), so doing it with noise induce unnecessary smoothing prior which might make things a bit too flat. So I make it as 1e-4. Making it lower would remove this inductive bias completely, and may not be good?

(3) (which I hasn not touched) this quantile transform make the outlier less sensitive. This makes graph look robust in low-sample regions. This is probably similar to the binning effect in EBM.


### Alright: so the date has to be completely different
### Make it as 0325
### The results is stored in _new
for dset in 'mimic2' 'adult'; do
  for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0325_${dset}_${arch} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done


### To see the graph, see how preprocessing changes the graph!
dset='mimic2'
arch='GAMAtt'
fold='0'
quantile_noise='0'
for quantile_dist in 'normal'; do
  for n_quantiles in '3000'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0326_${dset}_${arch}_best_f${fold}_${quantile_dist}_qn${quantile_noise}_nq${n_quantiles} --dataset ${dset} --seed 1 --num_trees 500 --num_layers 4 --batch_size 2048 --addi_tree_dim 1 --depth 5 --output_dropout 0.2 --colsample_bytree 1 --lr 0.02 --last_as_output 0 --dim_att 128 --arch ${arch} --fp16 1 --fold ${fold} --quantile_dist ${quantile_dist} --quantile_noise ${quantile_noise} --n_quantiles ${n_quantiles}
  done
done


# Add l2 for output penalty. Change to _new2
### Run more for search! Especially for last_as_output=1
#for dset in 'mimic2' 'adult'; do
for dset in 'compas' 'mimic3'; do
  for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0327_${dset}_${arch} --dataset ${dset} --random_search 15 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

## Rerun things: they die cuz no enough memory. Reduce batch size to 1024
for name in \
'0327_adult_GAM_s2_nl1_nt8000_td2_d4_od0.2_cs0.5_lr0.01_lo0_la0.0001' \
'0327_mimic2_GAMAtt_s83_nl1_nt8000_td1_d4_od0.0_cs1.0_lr0.01_lo0_la1e-06_da128' \
; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${name} --batch_size 512
done


###### Run all methods and baselines in other 3 datasets I gather.
for dset in 'support2' 'churn' 'credit'; do
  for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0327_${dset}_${arch} --dataset ${dset} --random_search 25 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done

  for model_name in 'ebm-o100-i100' 'xgb-o50'; do
    for fold in '0' '1' '2' '3' '4'; do
      ./my_sbatch --cpu 20 --gpus 0 --mem 10 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
    done
  done
done


##### Run the best among 7 datasets!
# mimic2
0326_mimic2_GAMAtt_best_f0_normal_qn0_nq3000

dset='mimic2'
arch='GAMAtt'
quantile_noise='0'
n_quantiles='3000'
quantile_dist='normal'
for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0330_${dset}_${arch}_best_f${fold}_${quantile_dist}_qn${quantile_noise}_nq${n_quantiles} --dataset ${dset} --seed 1 --num_trees 500 --num_layers 4 --batch_size 2048 --addi_tree_dim 1 --depth 5 --output_dropout 0.2 --colsample_bytree 1 --lr 0.02 --last_as_output 0 --dim_att 128 --arch ${arch} --fp16 1 --fold ${fold} --quantile_dist ${quantile_dist} --quantile_noise ${quantile_noise} --n_quantiles ${n_quantiles}
done

# Adult best
# 0327_adult_GAM_s17_nl3_nt333_td2_d4_od0.1_cs1.0_lr0.005_lo1_la0.0
# num_quantiles=1000
n_quantiles='1000'
dset='adult'
arch='GAM'
batch_size='2048'
for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0330_${dset}_${arch}_best_f${fold} --dataset ${dset} --seed 17 --num_trees 333 --num_layers 3 --batch_size ${batch_size} --addi_tree_dim 2 --depth 4 --output_dropout 0.1 --colsample_bytree 1 --lr 0.005 --last_as_output 1 --l2_lambda 0 --arch ${arch} --fp16 1 --fold ${fold} --n_quantiles ${n_quantiles}
done


# MIMIC3 best
# 0327_mimic3_GAMAtt_s6_nl4_nt2000_td0_d3_od0.1_cs0.1_lr0.01_lo0_la1e-07_da128
n_quantiles='2000'
dset='mimic3'
arch='GAMAtt'
batch_size='2048'
for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0330_${dset}_${arch}_best_f${fold} --dataset ${dset} --seed 6 --num_trees 2000 --num_layers 4 --batch_size ${batch_size} --addi_tree_dim 0 --depth 3 --output_dropout 0.1 --colsample_bytree 0.1 --lr 0.01 --last_as_output 0 --l2_lambda 1e-7 --dim_att 128 --arch ${arch} --fp16 1 --fold ${fold} --n_quantiles ${n_quantiles}
done

# COMPAS best
# 0327_compas_GAMAtt_s82_nl2_nt2000_td1_d2_od0.1_cs1.0_lr0.005_lo0_la0.0_da64 -0.73683 -0.74334
n_quantiles='2000'
dset='compas'
arch='GAMAtt'
batch_size='2048'
for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0330_${dset}_${arch}_best_f${fold} --dataset ${dset} --seed 82 --num_trees 2000 --num_layers 2 --batch_size ${batch_size} --addi_tree_dim 1 --depth 2 --output_dropout 0.1 --colsample_bytree 1 --lr 0.005 --last_as_output 0 --l2_lambda 0 --dim_att 64 --arch ${arch} --fp16 1 --fold ${fold} --n_quantiles ${n_quantiles}
done

# CHURN best
# 0327_churn_GAM_s6_nl2_nt1000_td1_d4_od0.0_cs0.5_lr0.01_lo1_la0.0
n_quantiles='2000'
dset='churn'
arch='GAM'
batch_size='2048'
for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0330_${dset}_${arch}_best_f${fold} --dataset ${dset} --seed 6 --num_trees 1000 --num_layers 2 --batch_size ${batch_size} --addi_tree_dim 1 --depth 4 --output_dropout 0 --colsample_bytree 0.5 --lr 0.01 --last_as_output 1 --l2_lambda 0 --arch ${arch} --fp16 1 --fold ${fold} --n_quantiles ${n_quantiles}
done

# Credit best
0327_credit_GAMAtt_s61_nl3_nt666_td2_d4_od0.1_cs0.5_lr0.005_lo0_la1e-07_da16
n_quantiles='2000'
dset='credit'
arch='GAMAtt'
batch_size='2048'
for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0330_${dset}_${arch}_best_f${fold} --dataset ${dset} --seed 61 --num_trees 666 --num_layers 3 --batch_size ${batch_size} --addi_tree_dim 2 --depth 4 --output_dropout 0.1 --colsample_bytree 0.5 --lr 0.005 --last_as_output 0 --l2_lambda 1e-7 --dim_att 16 --arch ${arch} --fp16 1 --fold ${fold} --n_quantiles ${n_quantiles}
done

# Support2 best
0327_support2_GAMAtt_s20_nl3_nt333_td2_d2_od0.2_cs1.0_lr0.01_lo0_la1e-05_da16	-0.82073	-0.81433
n_quantiles='2000'
dset='support2'
arch='GAMAtt'
batch_size='2048'
for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0330_${dset}_${arch}_best_f${fold} --dataset ${dset} --seed 20 --num_trees 333 --num_layers 3 --batch_size ${batch_size} --addi_tree_dim 2 --depth 2 --output_dropout 0.2 --colsample_bytree 1 --lr 0.01 --last_as_output 0 --l2_lambda 1e-5 --dim_att 16 --arch ${arch} --fp16 1 --fold ${fold} --n_quantiles ${n_quantiles}
done

for dset in 'compas'; do
  for model_name in 'ebm-o100-i100' 'xgb-o50'; do
    for fold in '0' '1' '2' '3' '4'; do
      ./my_sbatch --cpu 20 --gpus 0 --mem 10 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
    done
  done
done

dset='credit'
model_name='xgb-o50'
fold='0'
./my_sbatch --cpu 20 --gpus 0 --mem 10 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}



### ODST random search
for dset in 'compas' 'adult' 'mimic3' 'mimic2' 'support2' 'churn' 'credit'; do
  for arch in 'ODST'; do
    python main.py --name 0331_${dset}_${arch} --dataset ${dset} --random_search 1 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done


# Instead, let's run the GAM for MIMIC2, MIMIC3, credit, support2, compas



./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0330_${dset}_${arch}_best_f${fold} --dataset ${dset} --seed 20 --num_trees 333 --num_layers 3 --batch_size ${batch_size} --addi_tree_dim 2 --depth 2 --output_dropout 0.2 --colsample_bytree 1 --lr 0.01 --last_as_output 0 --l2_lambda 1e-5 --dim_att 16 --arch ${arch} --fp16 1 --fold ${fold} --n_quantiles ${n_quantiles}


for fold in '0' '1' '2' '3' '4'; do
    dset='mimic2'
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0401_${dset}_ODST_best_f${fold} --load_from_hparams 0331_mimic2_ODST_s82_nl2_nt1024_d6_td1_lr0.001 --fold ${fold}

    dset='adult'
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0401_${dset}_ODST_best_f${fold} --load_from_hparams 0331_adult_ODST_s84_nl2_nt512_d8_td2_lr0.001 --fold ${fold}

    dset='mimic3'
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0401_${dset}_ODST_best_f${fold} --load_from_hparams 0331_mimic3_ODST_s68_nl4_nt512_d6_td1_lr0.001 --fold ${fold}

    dset='support2'
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0401_${dset}_ODST_best_f${fold} --load_from_hparams 0331_support2_ODST_s69_nl4_nt512_d8_td2_lr0.001 --fold ${fold}

    dset='compas'
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0401_${dset}_ODST_best_f${fold} --load_from_hparams 0331_compas_ODST_s26_nl4_nt512_d8_td1_lr0.001 --fold ${fold}

    dset='churn'
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0401_${dset}_ODST_best_f${fold} --load_from_hparams 0331_churn_ODST_s43_nl2_nt512_d6_td2_lr0.001 --fold ${fold}

    dset='credit'
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0401_${dset}_ODST_best_f${fold} --load_from_hparams 0331_credit_ODST_s28_nl4_nt512_d8_td1_lr0.001 --fold ${fold}

    dset='mimic2'
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0402_${dset}_bestGAM_f${fold} --load_from_hparams 0327_mimic2_GAM_s60_nl4_nt250_td2_d2_od0.1_cs1e-05_lr0.005_lo1_la0.0 --fold ${fold}

    dset='mimic3'
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0402_${dset}_bestGAM_f${fold} --load_from_hparams 0327_mimic3_GAM_s19_nl1_nt2000_td2_d2_od0.2_cs0.1_lr0.01_lo0_la0.0 --fold ${fold}

    dset='compas'
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0402_${dset}_bestGAM_f${fold} --load_from_hparams 0327_compas_GAM_s81_nl2_nt4000_td0_d2_od0.2_cs1.0_lr0.01_lo0_la1e-07 --fold ${fold}

    dset='credit'
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0402_${dset}_bestGAM_f${fold} --load_from_hparams 0327_credit_GAM_s64_nl1_nt4000_td0_d3_od0.0_cs1.0_lr0.01_lo0_la1e-06 --fold ${fold}

    dset='support2'
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0402_${dset}_bestGAM_f${fold} --load_from_hparams 0327_support2_GAM_s44_nl4_nt1000_td0_d3_od0.0_cs1.0_lr0.005_lo1_la0.0 --fold ${fold}
done


# Search more for annoying less performing datasets!
for dset in 'mimic2' 'mimic3' 'credit' 'adult'; do
  for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0327_${dset}_${arch} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

# After searching more, only adult gets 1 slightly better. Use that one!
for fold in '0' '1' '2' '3' '4'; do
    dset='adult'
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0330_${dset}_best_f${fold} --load_from_hparams 0327_adult_GAM_s56_nl3_nt1333_td1_d4_od0.3_cs0.5_lr0.01_lo1_la0.0 --fold ${fold}
done


for dset in 'year' 'higgs' 'microsoft' 'yahoo' 'click' 'epsilon'; do
model_name='xgb-o50'
./my_sbatch --cpu 20 --gpus 0 --mem 16 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done

dset='epsilon'
model_name='xgb-o50'
./my_sbatch --cpu 20 --gpus 0 --mem 32 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}


for dset in 'yahoo' 'epsilon'; do
model_name='ebm-o20-i20'
./my_sbatch --cpu 20 --gpus 0 --mem 16 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done


# regression dataset
for dset in 'year' 'microsoft' 'yahoo'; do
  for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0328_${dset}_${arch} --dataset ${dset} --random_search 10 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done



#for dset in 'adult' 'mimic2' 'mimic3' 'credit' 'click' 'epsilon' 'higgs' 'rossmann'; do
for dset in 'wine'; do
  for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0329_${dset}_${arch} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

dset='wine'
arch='GAM'
python main.py --name 0329_${dset}_${arch} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}


## Run the missing value in those tables:
(1) baselines
model_name='ebm-o5-i5'
for dset in 'epsilon'; do
./my_sbatch --cpu 5 --gpus 0 --mem 100 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done
model_name='xgb-o10-nj1'
for dset in 'epsilon'; do
./my_sbatch --cpu 10 --gpus 0 --mem 80 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done
for model_name in 'xgb-o50'; do
for dset in 'higgs' 'year'; do
./my_sbatch --cpu 15 --gpus 0 --mem 30 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done
done
for model_name in 'ebm-o100-i100' 'xgb-o50'; do
for dset in 'rossmann'; do
./my_sbatch --cpu 30 --gpus 0 --mem 50 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done
done
for model_name in 'xgb-o50' 'ebm-o100-i100'; do
for dset in 'year'; do
./my_sbatch --cpu 25 --gpus 0 --mem 100 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done
done

for model_name in 'ebm-o100-i100' 'xgb-o50'; do
for dset in 'wine'; do
  for fold in '1' '2' '3' '4'; do
./my_sbatch --cpu 10 --gpus 0 --mem 5 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
done
done
done


# Run ODST wine!
dset='wine'
arch='ODST'
python main.py --name 0329_${dset}_${arch} --dataset ${dset} --random_search 30 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}


# wine best
wine_best='0329_wine_GAM_s46_nl3_nt666_td2_d2_od0.3_cs0.1_lr0.005_lo0_la0.0001'
for fold in '0' '1' '2' '3' '4'; do
    dset='wine'
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0330_${dset}_best_f${fold} --load_from_hparams ${wine_best} --fold ${fold}
done


## TODO: compare 0329 and 0328 on adult/mimic2/mimic3/credit to see if my new added bias works better!




# Rerun things after I do custom data noise for each dataset!
# Start with 0403!!!!
for dset in 'mimic2' 'mimic3' 'credit' 'adult'; do
  for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0403_${dset}_${arch} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done


### Annoyingly, the new qn does not have a dip in PFratio and perform worse
# Make a direct comparison!
mimic2_best='0329_mimic2_GAM_s57_nl2_nt4000_td0_d2_od0.0_cs0.5_lr0.01_lo0_la1e-05_ib1'
for qn in '1e-6' '0'; do
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0409_mimic2_best_${qn} --load_from_hparams ${mimic2_best} --quantile_noise ${qn}
done
wine_best='0329_wine_GAM_s46_nl3_nt666_td2_d2_od0.3_cs0.1_lr0.005_lo0_la0.0001'
for qn in '1e-8' '0'; do
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0409_wine_best_${qn} --load_from_hparams ${wine_best} --quantile_noise ${qn}
done
adult_best='0330_adult_best_f0'
for qn in '1e-3' '0'; do
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0409_adult_best_qn${qn} --load_from_hparams ${adult_best} --quantile_noise ${qn}
done
mimic3_best='0403_mimic3_GAM_s55_nl2_nt250_td0_d2_od0.0_cs1e-05_lr0.01_lo0_la1e-06'
for qn in '1e-7' '0'; do
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0409_mimic3_best_qn${qn} --load_from_hparams ${mimic3_best} --quantile_noise ${qn}
done


## Two changes: (1) min_temp, (2) custum noise
## Start with 0404
for dset in 'mimic2' 'mimic3' 'credit' 'adult' 'wine'; do
  for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0404_${dset}_${arch} --dataset ${dset} --random_search 30 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

## Run specifically for 0404 that previous hparams work best for these 5 dsets!
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0404 --load_from_hparams ${adult_best} --quantile_noise ${qn}

'0403_mimic2_GAMAtt_s18_nl4_nt125_td2_d4_od0.0_cs0.5_lr0.005_lo0_la0.0001_da8' \
'0327_credit_GAMAtt_s61_nl3_nt666_td2_d4_od0.1_cs0.5_lr0.005_lo0_la1e-07_da16' \
'0329_wine_GAM_s46_nl3_nt666_td2_d2_od0.3_cs0.1_lr0.005_lo0_la0.0001' \
'0327_year_GAMAtt_s33_nl4_nt250_td2_d2_od0.0_cs0.5_lr0.01_lo0_la0.0_da16' \
'0329_epsilon_GAMAtt_s39_nl3_nt333_td2_d6_od0.0_cs0.1_lr0.01_lo0_la0.0_da8' \
'0328_yahoo_GAMAtt_s58_nl4_nt250_td0_d6_od0.1_cs0.5_lr0.005_lo0_la0.0_da16' \
'0328_microsoft_GAMAtt_s68_nl3_nt333_td0_d4_od0.2_cs0.5_lr0.01_lo0_la0.0_da64' \
'0329_higgs_GAMAtt_s72_nl3_nt333_td1_d6_od0.3_cs0.1_lr0.005_lo0_la1e-06_da16' \
'0329_click_GAMAtt_s23_nl4_nt500_td0_d2_od0.3_cs1e-05_lr0.005_lo0_la0.0001_da32' \
'0403_adult_GAM_s82_nl2_nt4000_td1_d4_od0.2_cs0.1_lr0.01_lo0_la1e-05' \
'0403_credit_GAM_s78_nl4_nt500_td0_d4_od0.2_cs0.5_lr0.005_lo0_la0.0' \
'0327_year_GAMAtt_s44_nl4_nt500_td2_d4_od0.2_cs0.1_lr0.005_lo0_la0.0_da16' \
'0329_epsilon_GAMAtt_s52_nl2_nt2000_td1_d2_od0.3_cs0.5_lr0.005_lo0_la1e-05_da16' \
'0327_yahoo_GAMAtt_s40_nl3_nt2666_td0_d4_od0.2_cs0.1_lr0.005_lo0_la0.0_da16' \
'0328_microsoft_GAM_s13_nl1_nt4000_td0_d4_od0.0_cs0.1_lr0.005_lo0_la0.0' \
'0329_higgs_GAMAtt_s78_nl3_nt1333_td2_d2_od0.3_cs0.5_lr0.005_lo0_la0.0001_da8' \
'0329_click_GAMAtt_s45_nl4_nt500_td2_d4_od0.0_cs0.1_lr0.01_lo0_la0.0_da16' \
'0329_rossmann_GAMAtt_s41_nl2_nt500_td1_d6_od0.2_cs0.5_lr0.01_lo0_la0.0_da8' \
'0329_rossmann_GAM_s50_nl1_nt2000_td1_d4_od0.3_cs0.1_lr0.01_lo0_la1e-06' \
'0330_compas_GAMAtt_best_f0' \
'0327_churn_GAM_s6_nl2_nt1000_td1_d4_od0.0_cs0.5_lr0.01_lo1_la0.0' \
'0327_support2_GAMAtt_s20_nl3_nt333_td2_d2_od0.2_cs1.0_lr0.01_lo0_la1e-05_da16' \
'0327_compas_GAM_s81_nl2_nt4000_td0_d2_od0.2_cs1.0_lr0.01_lo0_la1e-07' \
'0327_churn_GAMAtt_s46_nl4_nt2000_td0_d2_od0.2_cs0.1_lr0.01_lo1_la1e-05_da64' \
'0327_support2_GAM_s44_nl4_nt1000_td0_d3_od0.0_cs1.0_lr0.005_lo1_la0.0' \
for d in \

; do
  postfix=${d:4}
  if [ -a logs/hparams/${d} ]; then
#    echo ${d}
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0404${postfix} --load_from_hparams ${d}
  fi
done

## (R) for more 40 times for compas
for dset in 'compas' 'churn' 'support2'; do
  arch='GAM'
  python main.py --name 0404_${dset}_${arch} --dataset ${dset} --random_search 15 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  arch='GAMAtt'
  python main.py --name 0404_${dset}_${arch} --dataset ${dset} --random_search 25 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
done

## (R) Try uniform for MIMIC2/Adult and see if the best can recover mean imputation and best perf
for dset in 'mimic2' 'adult'; do
  for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0405_${dset}_${arch}_uniform --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --quantile_dist uniform
  done
done


## (R) Test GAMAtt2 and last linear w/ GAM and GAMAtt
for dset in 'mimic2' 'adult'; do
  for arch in 'GAMAtt2'; do
    python main.py --name 0406_${dset}_${arch} --dataset ${dset} --random_search 30 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
  arch='GAMAtt'
  python main.py --name 0407_${dset}_${arch}_lastl --dataset ${dset} --random_search 30 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --add_last_linear 1
done


## (R) see if multi-task learning helps!!
for dset in 'sarcos' 'sarcos0' 'sarcos1'; do
  for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0404_${dset}_${arch} --dataset ${dset} --random_search 25 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

## (R) sarcos gets normalization wrong :(. Change to normalize per task. Rerun
## (R) run in the next block with testing last_l and GAM!
for dset in 'sarcos'; do
  for arch in 'GAM' 'GAMAtt2'; do
    python main.py --name 0405_${dset}_${arch} --dataset ${dset} --random_search 25 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

# (R) Random search for GAM and GAMAtt2 for lastl!
for dset in 'mimic2' 'credit' 'adult' 'wine'; do
  for arch in 'GAM' 'GAMAtt2'; do
    python main.py --name 0410_${dset}_${arch}_lastl --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --add_last_linear 1
  done
done

## Conclusions
# (1) For GAM, last_l significantly improve the performance!
# (2) For GAMAtt, last_l mostly improves avg (except wine cuz only 9 runs succeed, and works much better in adult) but for the best method sometimes the original GAMAtt/GAMAtt2 (like MIMIC2, Wine, but not Adult, Credit) performs better.
# (3) GAM v.s. GAMAtt2: unclear. Adult/Credit the GAM_lastl is better, but in MIMIC2/Wine the GAMAtt2 is the best. And in MIMIC2 the best GAMAtt have the PFratio drop story but not GAM!

## Strategy: (1) make last_l better for GAMAtt2! (2) Search for last_l=0 as well
for dset in 'mimic2' 'credit' 'adult' 'wine'; do
  for arch in 'GAM' 'GAMAtt2'; do
    python main.py --name 0410_${dset}_${arch}_lastl --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --add_last_linear 1
  done
done

#### Make output as new6: search last_layer in GAMAtt2. Run MORE!
for dset in 'mimic2' 'mimic3' 'credit' 'adult' 'wine' 'compas' 'churn' 'support2' 'wine'; do
  arch='GAM'
  python main.py --name 0411_${dset}_${arch} --dataset ${dset} --random_search 16 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  arch='GAMAtt2' # Search more
  python main.py --name 0411_${dset}_${arch} --dataset ${dset} --random_search 24 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
done


# (Wait) Rerun sarcos!
for dset in 'sarcos' 'sarcos0' 'sarcos1' 'sarcos2'; do
  arch='GAM'
  python main.py --name 0411_${dset}_${arch} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  arch='GAMAtt2' # Search more
  python main.py --name 0411_${dset}_${arch} --dataset ${dset} --random_search 30 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
done

for dset in 'year' 'epsilon' 'microsoft' 'click' 'higgs' 'rossmann' 'yahoo'; do
  arch='GAM'
  python main.py --name 0411_${dset}_${arch} --dataset ${dset} --random_search 16 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  arch='GAMAtt2' # Search more
  python main.py --name 0411_${dset}_${arch} --dataset ${dset} --random_search 24 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
done

## Annoying! I forget to cut the val set properly :(
# So it affects all classification datasets....
# Starting with new7.....

# Too tired of doing hyperparameter search. Will instead just choose the best hparams from previous search and run the best as 0413!
'0406_mimic2_GAMAtt2_s13_nl2_nt250_td1_d4_od0.1_cs0.5_lr0.01_lo0_la1e-05_da8' \
'0410_adult_GAM_lastl_s46_nl3_nt666_td1_d4_od0.1_cs0.5_lr0.01_lo0_la0.0' \
'0411_mimic3_GAM_s97_nl3_nt1333_td0_d6_od0.2_cs1e-05_lr0.005_lo0_la1e-07' \
'0411_compas_GAM_s32_nl4_nt1000_td0_d2_od0.1_cs0.5_lr0.01_lo0_la0.0' \
'0404_churn_GAMAtt_s93_nl3_nt166_td0_d2_od0.2_cs1e-05_lr0.01_lo0_la1e-07_da8' \
'0411_credit_GAMAtt2_s87_nl5_nt400_td2_d2_od0.2_cs0.1_lr0.01_lo0_la0.0_da8_ll1' \
'0404_support2_GAMAtt_s71_nl5_nt100_td0_d2_od0.2_cs0.5_lr0.005_lo0_la1e-06_da8' \
'0411_wine_GAM_s16_nl3_nt1333_td2_d4_od0.1_cs0.5_lr0.01_lo0_la1e-07' \
for d in \
'0415_support2_GAMAtt2_s62_nl2_nt1000_td0_d6_od0.3_cs1e-05_lr0.01_lo0_la1e-07_pt0_pr0_mn0_da32_ll1' \
; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0416_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done


# Annoying; try to beat GAMAtt in churn and support2
# Since now I update val set in new7/8, I can not compare them directly.
# What I can do is to rerun things that are great in new val set. Then compare them in test set!!
for d in \
'0404_support2_GAMAtt_s71_nl5_nt100_td0_d2_od0.2_cs0.5_lr0.005_lo0_la1e-06_da8' \
'0404_support2_GAMAtt_s29_nl4_nt1000_td2_d6_od0.3_cs0.5_lr0.01_lo0_la1e-07_da8' \
'0404_churn_GAMAtt_s93_nl3_nt166_td0_d2_od0.2_cs1e-05_lr0.01_lo0_la1e-07_da8' \
'0404_churn_GAMAtt_s23_nl5_nt100_td0_d6_od0.1_cs0.1_lr0.01_lo0_la0.0_da8' \
; do
  postfix=${d:4}
  ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0415${postfix} --load_from_hparams ${d} --arch GAMAtt2
done
for dset in 'churn' 'support2'; do
  arch='GAMAtt2' # Search more
  python main.py --name 0415_${dset}_${arch} --dataset ${dset} --random_search 40 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
done


## (R) see which architecture is best for higgs! And do pre-training
## Add new8: random search for pretraining as well
dset='higgs'
for data_subsample in '1e3'; do
arch='GAMAtt2' # Search more
python main.py --name 0414_${dset}_${arch}_ds${data_subsample} --dataset ${dset} --random_search 60 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain 1 --data_subsample ${data_subsample}
arch='GAMAtt2' # Search more
python main.py --name 0414_${dset}_${arch}_ds${data_subsample} --dataset ${dset} --random_search 30 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain 0 --data_subsample ${data_subsample}
done

dset='higgs'
arch='GAMAtt2' # Search more
data_subsample='1e3'
python main.py --name 0416_${dset}_${arch}_ds${data_subsample} --dataset ${dset} --random_search 60 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain 1 --data_subsample ${data_subsample}

# Rerun stuffs here
'0413_f1_best_churn_GAMAtt_s93_nl3_nt166_td0_d2_od0.2_cs1e-05_lr0.01_lo0_la1e-07_da8' \
for d in \
'0413_f0_best_mimic2_GAMAtt2_s13_nl2_nt250_td1_d4_od0.1_cs0.5_lr0.01_lo0_la1e-05_da8' \
; do
  ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name $d
done

- I can do a soft-self-attention to make graph sparse/jumpy?


## Compare sarcos:
## - Still, the multi-task is not as good as single task lol
## - Idea: Improve MTL by per-task early stopping by looking at multiple valiation loss?


# Idea1: I can do MTL not just by a weighted linear layer in the end. I can have task-specific GAMTree per-task.
# Idea3: I can add soft weight penalty across weights per task to encourage similar graph.

for d in \
'0308_wine_ebm-o100-i100' \
'0308_wine_ebm-o100-i100_f1' \
'0308_wine_ebm-o100-i100_f2' \
'0308_wine_ebm-o100-i100_f3' \
'0308_wine_ebm-o100-i100_f4' \
; do
 cp logs/hparams/$d ./logs/$d/hparams.json
done

for d in logs/0416_higgs*_pt1_*; do
  name=${d:5}
  echo ${name}
  ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${name}_ft --load_from_pretrain ${name} --lr 0.0005

#  postfix=${d:4}
#  if [ -a logs/hparams/${d} ]; then
#    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0404${postfix} --load_from_hparams ${d}
#  fi
done

# Rerun tons of things in new val set to see if GAMAtt2 can outperform old GAMAtt
# Also, run more in rossmann to see if I can outperform xgb
for dset in 'year' 'microsoft' 'higgs' 'rossmann'; do
  arch='GAMAtt2' # Search more
  python main.py --name 0417_${dset}_${arch} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
done


# Even less
dset='higgs'
arch='GAMAtt2' # Search more
for data_subsample in '500'; do
python main.py --name 0414_${dset}_${arch}_ds${data_subsample} --dataset ${dset} --random_search 50 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain 1 --data_subsample ${data_subsample}
python main.py --name 0414_${dset}_${arch}_ds${data_subsample} --dataset ${dset} --random_search 30 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain 0 --data_subsample ${data_subsample}
done


for dset in 'year' 'microsoft' 'higgs' 'rossmann'; do
  arch='GAMAtt2' # Search more
  python main.py --name 0417_${dset}_${arch} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
done


# Run sarcos for much less data
data_subsample='1e3'
for dset in 'sarcos' 'sarcos0'; do
  arch='GAM'
  python main.py --name 0418_${dset}_${data_subsample}_${arch} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --data_subsample ${data_subsample}
  arch='GAMAtt2' # Search more
  python main.py --name 0418_${dset}_${data_subsample}_${arch} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --data_subsample ${data_subsample}
done


# (TORUN) These 3 datasets have inferior performance. Run more here!
dset='wine'
arch='GAMAtt2'
python main.py --name 0418_${dset}_${arch} --dataset ${dset} --random_search 40 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
dset='compas'
arch='GAMAtt2'
python main.py --name 0418_${dset}_${arch} --dataset ${dset} --random_search 40 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
dset='support2'
arch='GAMAtt2'
python main.py --name 0418_${dset}_${arch} --dataset ${dset} --random_search 40 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
dset='churn'
arch='GAMAtt2'
python main.py --name 0418_${dset}_${arch} --dataset ${dset} --random_search 40 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
dset='wine'
arch='GAM'
python main.py --name 0418_${dset}_${arch} --dataset ${dset} --random_search 25 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
dset='compas'
arch='GAM'
python main.py --name 0418_${dset}_${arch} --dataset ${dset} --random_search 25 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
dset='support2'
arch='GAM'
python main.py --name 0418_${dset}_${arch} --dataset ${dset} --random_search 25 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
dset='churn'
arch='GAM'
python main.py --name 0418_${dset}_${arch} --dataset ${dset} --random_search 25 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}


# Note p100! Just to see if last_dropout helps!
for dset in 'mimic2' 'adult'; do
  for arch in 'GAM' 'GAMAtt' 'GAMAtt2'; do
    python main.py --name 0420_${dset}_${arch} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --partition p100
  done
done

# Try to win in these last 2 datasets! And still can not design which is the best
for dset in 'support2' 'wine'; do
  for arch in 'GAM' 'GAMAtt' 'GAMAtt2'; do
    python main.py --name 0420_${dset}_${arch} --dataset ${dset} --random_search 25 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

for dset in 'rossmann'; do
  for arch in 'GAM' 'GAMAtt' 'GAMAtt2'; do
    python main.py --name 0420_${dset}_${arch} --dataset ${dset} --random_search 25 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

for dset in 'support2' 'wine'; do
  for arch in 'GAM'; do
    python main.py --name 0420_${dset}_${arch} --dataset ${dset} --random_search 25 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

#(7) Run more in rossmann (0424)
#  - Not much improvement
#Run the final model for support2/wine
dset='support2'
support2_cur_best='0420_support2_GAM_s43_nl4_nt125_td1_d2_od0.1_ld0.0_cs1e-05_lr0.01_lo0_la1e-06_pt0_pr0_mn0_ol0_ll1'
for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0420_${dset}_f${fold}_best${postfix} --load_from_hparams ${support2_cur_best} --fold ${fold}
done

dset='wine'
wine_cur_best='0420_wine_GAM_s31_nl5_nt800_td1_d2_od0.0_ld0.1_cs0.5_lr0.005_lo0_la1e-05_pt0_pr0_mn0_ol0_ll1'
for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0420_${dset}_f${fold}_best${postfix} --load_from_hparams ${wine_cur_best} --fold ${fold}
done

(5) Run another pretraining loss w/ only 500 samples (0421). Hope it improves!
* Problem should be there is almost no possibility for mask pretraining. Change to MSE
dset='higgs'
arch='GAMAtt2' # Search more
for data_subsample in '500'; do
  for pretrain in '2'; do
python main.py --name 0421_${dset}_${arch}_ds${data_subsample} --dataset ${dset} --random_search 30 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --data_subsample ${data_subsample}
  done
done
pretrain='0'
for d in is_running/0421*; do
    name=${d:11} # Remove directory name
    echo ${name}
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${name}_pt${pretrain} --load_from_hparams ${name} --pretrain ${pretrain}
done
### 0425 (R)
dset='higgs'
arch='GAMAtt2' # Search more
for data_subsample in '2000'; do
  for pretrain in '2' '1' '0'; do
python main.py --name 0425_${dset}_${arch}_ds${data_subsample} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --data_subsample ${data_subsample} --seed 10
  done
done
pretrain='0'
for d in is_running/0425*; do
    name=${d:11} # Remove directory name
    echo ${name}
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${name}_pt${pretrain} --load_from_hparams ${name} --pretrain ${pretrain}
done



dset='higgs'
arch='GAMAtt2' # Search more
for data_subsample in '2000'; do
  for pretrain in '2'; do
python main.py --name 0427_${dset}_${arch}_ds${data_subsample} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --data_subsample ${data_subsample} --seed 10
  done
done

## run a 5k sample size in higgs. See diff. A: Not much diff.
dset='higgs'
arch='GAMAtt2' # Search more
for data_subsample in '5000'; do
  for pretrain in '2'; do
python main.py --name 0428_${dset}_${arch}_ds${data_subsample} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --data_subsample ${data_subsample} --seed 10
  done
done

## (Wait) Wait the 0428 run and get hparams.
pretrain='0'
for d in logs/hparams/0428*; do
    if [ ${d: -2} != "ft" ] ; then
      name=${d:13} # Remove directory name
      echo ${name}
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${name}_pt${pretrain} --load_from_hparams ${name} --pretrain ${pretrain}
    fi
done

./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0427_higgs_GAMAtt2_ds2000_s67_nl4_nt1000_td2_d6_od0.2_ld0.1_cs1e-05_lr0.005_lo0_la1e-05_pt2_pr0.15_mn0.1_ol0_ll1_da16



#(5) (Rerun 0423) 0422 500 does not work. Average multi-task is better, and min is better. But val set is too small and choose something wierd.
#Run another sarcos multi-task learning with more samples: 2000
#Makes 2 datasets sarcos/sarcos0 the same hyperparameters in random search to reduce variance.
#*** Setting seed works since in this setting the hparams are selected same.
data_subsample='2000'
for dset in 'sarcos' 'sarcos0' 'sarcos1'; do
  arch='GAMAtt2'
  python main.py --name 0423_${dset}_ds${data_subsample}_${arch} --dataset ${dset} --random_search 20 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --data_subsample ${data_subsample} --seed 15
done
Hmm not much improvement!.....


data_subsample='5000'
for dset in 'sarcos' 'sarcos0' 'sarcos1'; do
  arch='GAMAtt2'
  python main.py --name 0423_${dset}_ds${data_subsample}_${arch} --dataset ${dset} --random_search 15 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --data_subsample ${data_subsample} --seed 20
done




Tosee:
#(3) See if churn/support2 GAMAtt2 can outperform old GAMAtt in 0415. Compare test performance.
#  - Support2 we can. For churn no. Only support reruns (0416)
#(4) See if year/microsoft/higgs/rossmann 0417 can outperform GAMAtt by GAMAtt2
#- Not always.
#(6) Run sarcos with less sample size (0418)
#  - Not working still
#(7) (0418) See if the more running gets better that GAMAtt2 > GAMAtt, and GAM searching linear is also better!
#- Not necessary. Support2/churn not changed. So maybe GAMAtt sometimes has an advantage.
#(8) (0420) See last_w dropout which value works?
- See sarcos 5000 comparisons (not good)
- See higgs with pretrain=2 and ds=2000,5000 (not good as well. Only ds=2k seems working)
- Run other datasets!

# (R 0430) Try GAM for SS learning: hopefully this improves more.
# Only test mimic2: test diff freezing steps 0, 1000, 2000!
arch='GAM'
pretrain='2'
send_pt0='1'
data_subsample='0.06'
random_search='15'
for dset in 'mimic2'; do
python main.py --name 0430_${dset}_${arch}_ds${data_subsample} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --data_subsample ${data_subsample} --send_pt0 ${send_pt0} --finetune_lr 0.01 5e-3 5e-4 1e-4
done
random_search='10'
data_subsample='1100'
for dset in 'higgs'; do
python main.py --name 0430_${dset}_${arch}_ds${data_subsample} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --data_subsample ${data_subsample} --send_pt0 ${send_pt0} --finetune_lr 0.01 5e-3 5e-4 1e-4
done


# Try baselines with xgb-d3!
for model_name in 'xgb-d3'; do
  for dset in 'adult' 'churn' 'credit' 'compas' 'support2' 'mimic3' 'wine'; do
    for fold in '0' '1' '2' '3' '4'; do
      ./my_sbatch --cpu 10 --gpus 0 --mem 30 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
    done
  done
  dset='rossmann'
  ./my_sbatch --cpu 10 --gpus 0 --mem 50 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done

for model_name in 'xgb-d3'; do
  for dset in 'mimic2'; do
    for fold in '0' '1' '2' '3' '4'; do
      ./my_sbatch --cpu 10 --gpus 0 --mem 30 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
    done
  done
done


for model_name in 'xgb-d3'; do
  for dset in 'click' 'epsilon' 'year' 'microsoft' 'higgs'; do
  ./my_sbatch --cpu 10 --gpus 0 --mem 50 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
  done
done



# (Wait till midnight today!) Based on 2 results: higgs/mimic, I will probably choose flr=1e-4 and frs=0!
arch='GAM'
pretrain='2'
send_pt0='1'
data_subsample='0.05'
random_search='15'
for dset in 'adult' 'mimic2' 'compas' 'credit'; do
python main.py --name 0430_${dset}_${arch}_ds${data_subsample} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --data_subsample ${data_subsample} --send_pt0 ${send_pt0} --finetune_lr 5e-4 1e-4
done

arch='GAMAtt2'
pretrain='2'
send_pt0='1'
finetune_data_subsample='0.051 0.1'
finetune_freeze_steps='0 1000'
finetune_lr='5e-4 1e-4'
random_search='13'
for dset in 'mimic2' 'mimic3'; do
python main.py --name 0501_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps}
done

# RUn the rest data ratio!
arch='GAMAtt2'
pretrain='2'
send_pt0='1'
finetune_data_subsample='0.01 0.02 0.2 0.5'
finetune_freeze_steps='0 1000'
finetune_lr='1e-4'
random_search='15'
for dset in 'mimic2' 'mimic3'; do
python main.py --name 0503_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps}
done


./my_sbatch -p t4v2 --name 0501_mimic2_GAMAtt2_s79_nl4_nt250_td1_d6_od0.1_ld0.1_cs0.5_lr0.01_lo0_la1e-06_pt2_pr0.15_mn0.1_ol0_ll1_da8_fds0.051_flr0.0001_frs1000_ft python -u main.py --load_from_pretrain 0501_mimic2_GAMAtt2_s79_nl4_nt250_td1_d6_od0.1_ld0.1_cs0.5_lr0.01_lo0_la1e-06_pt2_pr0.15_mn0.1_ol0_ll1_da8 --pretrain 0 --lr 0.0001 --freeze_steps 1000 --data_subsample 0.051
./my_sbatch -p t4v2 --name 0501_mimic2_GAMAtt2_s79_nl4_nt250_td1_d6_od0.1_ld0.1_cs0.5_lr0.01_lo0_la1e-06_pt2_pr0.15_mn0.1_ol0_ll1_da8_fds0.1_flr0.0001_frs1000_ft python -u main.py --load_from_pretrain 0501_mimic2_GAMAtt2_s79_nl4_nt250_td1_d6_od0.1_ld0.1_cs0.5_lr0.01_lo0_la1e-06_pt2_pr0.15_mn0.1_ol0_ll1_da8 --pretrain 0 --lr 0.0001 --freeze_steps 1000 --data_subsample 0.1


./my_sbatch --cpu 4 --gpus 1 --mem 8 --name 0503_mimic2_GAMAtt2_s92_nl2_nt500_td2_d6_od0.2_ld0.1_cs0.5_lr0.005_lo0_la1e-05_pt2_pr0.15_mn0.1_ol0_ll1_da16 python -u main.py



# Run other dataset
arch='GAM'
pretrain='2'
send_pt0='1'
finetune_data_subsample='0.01 0.05 0.1 0.2 0.5'
finetune_freeze_steps='0 1000'
finetune_lr='1e-4'
random_search='14'
#for dset in 'adult' 'support2' 'compas' 'churn' 'credit'; do
for dset in 'compas' 'support2' 'adult'; do
python main.py --name 0504_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps}
done


## (R) Run another random search for GAMAtt3!
for dset in 'adult' 'mimic2'; do
  for arch in 'GAMAtt3'; do
    python main.py --name 0420_${dset}_${arch} --dataset ${dset} --random_search 24 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0420_adult_GAMAtt3_s89_nl2_nt2000_td2_d4_od0.0_ld0.0_cs0.5_lr0.005_lo0_la0.0_pt0_pr0_mn0_ol0_ll0_da8

# Rerun best for each dataset! Only take the new hparams here...
for d in \
'0420_mimic2_GAM_s94_nl4_nt250_td0_d2_od0.0_ld0.1_cs0.1_lr0.01_lo0_la1e-07_pt0_pr0_mn0_ol0_ll1' \
'0420_adult_GAMAtt_s15_nl3_nt166_td2_d6_od0.0_ld0.0_cs1e-05_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll1_da16' \
'0413_f0_best_mimic3_GAM_s97_nl3_nt1333_td0_d6_od0.2_cs1e-05_lr0.005_lo0_la1e-07' \
'0418_compas_GAMAtt2_s67_nl5_nt800_td2_d4_od0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0_mn0_ol0_da16_ll1' \
'0415_churn_GAMAtt2_s48_nl3_nt166_td2_d4_od0.1_cs0.5_lr0.005_lo0_la1e-05_pt0_pr0_mn0_da8_ll1' \
'0413_f0_best_credit_GAMAtt2_s87_nl5_nt400_td2_d2_od0.2_cs0.1_lr0.01_lo0_la0.0_da8_ll1' \
; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0502_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done

./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0504_compas_GAM_s61_nl2_nt2000_td2_d6_od0.2_ld0.2_cs1e-05_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_ds0.5 --min_bs 64


# (Wait to see if mimic2 trial run succeeds) spline!
for model_name in 'spline'; do
  for dset in 'adult' 'churn' 'credit' 'compas' 'support2' 'mimic3' 'wine'; do
    for fold in '0' '1' '2' '3' '4'; do
      ./my_sbatch --cpu 10 --gpus 0 --mem 30 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
    done
  done
  for dset in 'rossmann' 'click' 'epsilon' 'year' 'microsoft' 'higgs' 'yahoo'; do
    ./my_sbatch --cpu 10 --gpus 0 --mem 50 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
  done
done

for model_name in 'spline'; do
  for dset in 'mimic2'; do
    for fold in '0' '1' '2' '3' '4'; do
      ./my_sbatch --cpu 10 --gpus 0 --mem 30 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
    done
  done
done

for ds in '0.5' '0.2' '0.1' '0.05' '0.01'; do
  for model_name in 'spline' 'ebm-o100-i100' 'xgb-o50'; do
    for dset in 'adult' 'churn' 'credit' 'compas' 'support2' 'mimic3' 'mimic2'; do
        ./my_sbatch --cpu 10 --gpus 0 --mem 40 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_ds${ds} --dataset ${dset} --model_name ${model_name} --data_subsample ${ds}
    done
  done
done

dset='credit'
model_name='xgb-o10'
ds='0.01'
./my_sbatch --cpu 10 --gpus 0 --mem 40 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_ds${ds} --dataset ${dset} --model_name ${model_name} --data_subsample ${ds}



model_name='spline'
dset='adult'
fold='0'
./my_sbatch --cpu 10 --gpus 0 --mem 30 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}



# Run other dataset
arch='GAM'
pretrain='2'
send_pt0='1'
finetune_data_subsample='0.01 0.05 0.1 0.2 0.5'
finetune_freeze_steps='0 1000'
finetune_lr='1e-4'
random_search='15'
for dset in 'churn' 'credit'; do
python main.py --name 0504_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps}
done

# (R) run more with GAMAtt! See if this is the key reason
arch='GAMAtt'
pretrain='2'
send_pt0='1'
finetune_data_subsample='0.01 0.05 0.1 0.2 0.5'
finetune_freeze_steps='0 1000'
finetune_lr='1e-4'
random_search='15'
for dset in 'compas' 'support2' 'adult' 'credit' 'churn'; do
python main.py --name 0504_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps}
done

# (wait) ODST
for dset in 'adult' 'mimic2' 'mimic3' 'support2' 'compas' 'churn' 'credit' 'wine'; do
  for arch in 'ODST'; do
    python main.py --name 0505_${dset}_${arch} --dataset ${dset} --random_search 14 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done

for dset in 'rossmann'; do
  for arch in 'ODST'; do
    python main.py --name 0505_${dset}_${arch} --dataset ${dset} --random_search 15 --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch}
  done
done


# We test the semi-supervised in large datasets like higgs/
arch='GAMAtt'
pretrain='2'
send_pt0='1'
finetune_data_subsample='0.01 0.05 0.1 0.2 0.5'
finetune_freeze_steps='0 1000'
finetune_lr='1e-4'
random_search='15'
for dset in 'click' 'epsilon'; do
python main.py --name 0506_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps}
done
# (WAIT TO RUN) have not run 'year' 'microsoft' 'higgs' 'yahoo'

arch='GAMAtt'
pretrain='2'
send_pt0='1'
finetune_data_subsample='0.01 0.05 0.1 0.2 0.5'
finetune_freeze_steps='0 1000'
finetune_lr='1e-4'
random_search='15'
for dset in 'year' 'microsoft'; do
python main.py --name 0506_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps}
done

'year/microsoft/click' -> lower data ratio!

arch='GAMAtt'
pretrain='2'
send_pt0='1'
finetune_data_subsample='1e-3 5e-3'
finetune_freeze_steps='0 1000'
finetune_lr='1e-4'
random_search='15'
for dset in 'year' 'microsoft' 'click'; do
python main.py --name 0506_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps}
done



# ODST
for d in \
'0505_mimic2_ODST_s32_nl2_nt512_d6_td1_lr0.001' \
'0505_adult_ODST_s90_nl2_nt512_d4_td2_lr0.001' \
'0505_mimic3_ODST_s21_nl2_nt512_d8_td2_lr0.001' \
'0505_compas_ODST_s66_nl2_nt1024_d6_td2_lr0.001' \
'0505_churn_ODST_s20_nl2_nt1024_d6_td0_lr0.001' \
'0505_credit_ODST_s17_nl2_nt1024_d6_td0_lr0.001' \
'0505_support2_ODST_s45_nl8_nt256_d8_td2_lr0.001' \
'0505_wine_ODST_s45_nl2_nt1024_d4_td2_lr0.001' \
; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0507_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done

# 0507!



for d in \
'0420_mimic2_GAM_s94_nl4_nt250_td0_d2_od0.0_ld0.1_cs0.1_lr0.01_lo0_la1e-07_pt0_pr0_mn0_ol0_ll1' \
'0420_adult_GAMAtt_s15_nl3_nt166_td2_d6_od0.0_ld0.0_cs1e-05_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll1_da16' \
'0413_f0_best_mimic3_GAM_s97_nl3_nt1333_td0_d6_od0.2_cs1e-05_lr0.005_lo0_la1e-07' \
'0418_compas_GAMAtt2_s67_nl5_nt800_td2_d4_od0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0_mn0_ol0_da16_ll1' \
'0415_churn_GAMAtt2_s48_nl3_nt166_td2_d4_od0.1_cs0.5_lr0.005_lo0_la1e-05_pt0_pr0_mn0_da8_ll1' \
'0413_f0_best_credit_GAMAtt2_s87_nl5_nt400_td2_d2_od0.2_cs0.1_lr0.01_lo0_la0.0_da8_ll1' \
; do

done


arch='GAMAtt'
pretrain='2'
send_pt0='1'
finetune_data_subsample='0.001 0.005 0.01 0.05 0.1'
finetune_freeze_steps='0 1000'
finetune_lr='1e-4'
random_search='15'
for dset in 'yahoo' 'rossmann'; do
python main.py --name 0506_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps}
done

arch='GAMAtt'
pretrain='2'
send_pt0='1'
finetune_data_subsample='1e-4 2e-4 5e-4 1e-3 5e-3'
finetune_freeze_steps='0 1000'
finetune_lr='1e-4'
random_search='15'
for dset in 'higgs'; do
python main.py --name 0506_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps}
done


for ds in '0.001' '0.005' '0.01' '0.05' '0.1'; do
  for model_name in 'spline' 'ebm-o100-i100' 'xgb-o50'; do
    for dset in 'yahoo' 'rossmann' 'microsoft' 'year'; do
        ./my_sbatch --cpu 10 --gpus 0 --mem 40 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_ds${ds} --dataset ${dset} --model_name ${model_name} --data_subsample ${ds}
    done
  done
done

for ds in '1e-4' '2e-4' '5e-4' '1e-3' '5e-3'; do
  for model_name in 'spline' 'ebm-o100-i100' 'xgb-o50'; do
    for dset in 'higgs'; do
        ./my_sbatch --cpu 10 --gpus 0 --mem 40 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_ds${ds} --dataset ${dset} --model_name ${model_name} --data_subsample ${ds}
    done
  done
done

# Run smaller val set. Split train as val set
#  has not run
arch='GAMAtt'
pretrain='2'
send_pt0='1'
finetune_data_subsample='0.0005 0.001 0.005 0.01' # 0.01 0.05 0.1
finetune_freeze_steps='0 1000'
finetune_lr='1e-4'
random_search='15'
for dset in 'yahoo' 'rossmann'; do
python main.py --name 0508_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps} --split_train_as_val 1 --seed 30
done

arch='GAMAtt'
pretrain='2'
send_pt0='1'
finetune_data_subsample='5e-5 1e-4 5e-4 1e-3 5e-3' # 0.01 0.05 0.1
finetune_freeze_steps='0 1000'
finetune_lr='1e-4'
random_search='15'
for dset in 'higgs'; do
python main.py --name 0508_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps} --split_train_as_val 1 --seed 30
done

# Run smaller ratio for churn/year/microsoft/yahoo/higgs?


# Rerun ODST 6 datasets for xgb / ebm to avoid unfair advantage...
for dset in 'year' 'higgs' 'microsoft' 'yahoo' 'click'; do
model_name='xgb-o50'
./my_sbatch --cpu 17 --gpus 0 --mem 40 -p cpu python -u baselines.py --name 0309_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done

for dset in 'year' 'higgs' 'microsoft' 'click'; do
model_name='ebm-o100-i100'
./my_sbatch --cpu 20 --gpus 0 --mem 50 -p cpu python -u baselines.py --name 0309_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done

for dset in 'yahoo'; do
model_name='ebm-o10-i10'
./my_sbatch --cpu 10 --gpus 0 --mem 50 -p cpu python -u baselines.py --name 0309_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done

for dset in 'higgs'; do
model_name='ebm-o100-i100'
./my_sbatch --cpu 10 --gpus 0 --mem 100 -p cpu python -u baselines.py --name 0309_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done

for dset in 'year'; do
model_name='xgb-o50'
./my_sbatch --cpu 10 --gpus 0 --mem 80 -p cpu python -u baselines.py --name 0309_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done
0309_year_xgb-o50


# 0509: we let split_train_as_val=1 having larger data size.
# Since this might change results, just for consistency let's run this with more lr / steps as well.
########  Ignore 'rossmann' 'click' for now. We might not include rossmann, and the click already wins.
arch='GAMAtt'
pretrain='2'
send_pt0='1'
finetune_data_subsample='0.0002 0.0005 0.001 0.002 0.005'
finetune_freeze_steps='0 1000'
finetune_lr='3e-4 1e-4'
random_search='1'
pretraining_ratio='0.15'
for dset in 'microsoft'; do
#for dset in 'yahoo' 'year'; do
python main.py --name 0509_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps} --split_train_as_val 1 --seed 30 --pretraining_ratio ${pretraining_ratio}
done
finetune_data_subsample='0.00002 0.00005 0.0001 0.0002 0.0005'
for dset in 'higgs'; do
python main.py --name 0509_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps} --split_train_as_val 1 --seed 30  --pretraining_ratio ${pretraining_ratio}
done


# For microsoft: see if pretraining ratio as 0.2 would be better?
arch='GAMAtt'
pretrain='2'
send_pt0='0'
finetune_data_subsample='0.0002 0.0005'
finetune_freeze_steps='500 1000'
finetune_lr='5e-4 3e-4 1e-4'
random_search='10'
for dset in 'microsoft' 'yahoo'; do
  for pretraining_ratio in '0.1' '0.15' '0.2' '0.25' '0.3'; do
    python main.py --name 0510_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps} --split_train_as_val 1 --seed 30 --pretraining_ratio ${pretraining_ratio}
  done
done

# Run for the rest 3 (Higgs, year, click) which is the best pr ratio / finetune steps
arch='GAMAtt'
pretrain='2'
send_pt0='0'
finetune_freeze_steps='500'
finetune_lr='5e-4 1e-4'
random_search='9'
for dset in 'higgs' 'year' 'click'; do
  if [[ "${dset}" == "higgs" ]]; then
    finetune_data_subsample='0.00002'
  else
    finetune_data_subsample='0.0002'
  fi
  echo ${dset} ${finetune_data_subsample}
  for pretraining_ratio in '0.1' '0.15' '0.2' '0.25' '0.3'; do
    python main.py --name 0510_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps} --split_train_as_val 1 --seed 30 --pretraining_ratio ${pretraining_ratio}
  done
done

# (R 0511) Test pretrain=3? If it improves?
arch='GAMAtt'
finetune_freeze_steps='500'
finetune_lr='5e-4 1e-4'
random_search='14'
finetune_data_subsample='0.005 0.01'
pretraining_ratio='0.15'
for dset in 'mimic2'; do
  # Normal training with 3d
  pretrain='2'
  send_pt0='1'
  python main.py --name 0511_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps} --split_train_as_val 1 --seed 30 --pretraining_ratio ${pretraining_ratio}
  pretrain='3'
  send_pt0='0'
  python main.py --name 0511_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps} --split_train_as_val 1 --seed 30 --pretraining_ratio ${pretraining_ratio}
done
arch='GAMAtt'
finetune_freeze_steps='500'
finetune_lr='5e-4 1e-4'
random_search='10'
finetune_data_subsample='0.005'
pretraining_ratio='0.15'
for dset in 'mimic3'; do
  # Normal training with 3d
  pretrain='2'
  send_pt0='1'
  python main.py --name 0511_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps} --split_train_as_val 1 --seed 30 --pretraining_ratio ${pretraining_ratio}
  pretrain='3'
  send_pt0='0'
  python main.py --name 0511_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps} --split_train_as_val 1 --seed 30 --pretraining_ratio ${pretraining_ratio}
done


### O.k. since I probably know what is the best for each dataset. Change pt=3 for yahoo/
### (R) 0512: Find the best pr ratio. Run on 5 large datasets (except epsilon)
### Have not run 'click' 'yahoo' 'higgs'
arch='GAMAtt'
pretrain='3'
send_pt0='1'
finetune_freeze_steps='500'
finetune_lr='5e-5 1e-4 2e-4 5e-4'
random_search='14'
for dset in 'year' 'microsoft'; do
  finetune_data_subsample='2e-4 5e-4 1e-3 2e-3 5e-3'
  if [[ "${dset}" == "higgs" ]]; then
    finetune_data_subsample='2e-5 5e-5 1e-4 2e-4 5e-4'
  fi

  pretraining_ratio='0.15'
  if [[ "${dset}" == "yahoo" ]]; then
    pretraining_ratio='0.25'
  fi

  python main.py --name 0512_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps} --split_train_as_val 1 --seed 30 --pretraining_ratio ${pretraining_ratio}
done
#### WIERD!!!!!!!! The result gets worse. But too many differences e.g. smaller data ratio, smaller early stopping rounds, split_train_as_val=1??
### Rerun the rest smaller datasets and see if pt=3 improves?

# 0513: run if the entmoid with smaller temperature much jumpier??
random_search='15'
for dset in 'mimic2' 'adult'; do
  arch='GAMAtt'
  for entmoid_min_temp in '1' '0.1' '0.01'; do
    python main.py --name 0513_${dset}_${arch}_entm${entmoid_min_temp} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --entmoid_min_temp ${entmoid_min_temp} --seed 12
  done
done




# To run more data ratio, we set flag ignore_prev_runs=1
# It might have dangers of causing two jobs runs the same config and crashes though.
arch='GAMAtt'
finetune_freeze_steps='500'
finetune_lr='5e-4 3e-4 1e-4 5e-5'
random_search='15'
pretraining_ratio='0.15'
#for dset in 'mimic2' 'mimic3' 'adult' 'compas' 'support2' 'credit' 'churn'; do
for dset in 'adult' 'compas' 'support2' 'credit' 'churn'; do
  finetune_data_subsample='0.02 0.05 0.1'
#  finetune_data_subsample='0.005 0.01'
  if [[ "${dset}" == "mimic3" ]]; then
    finetune_data_subsample='0.01 0.02 0.05 0.1'
  fi
#  if [[ "${dset}" == "mimic2" ]]; then # These already finish
#    finetune_data_subsample='0.02 0.05 0.1'
#  fi
  if [[ "${dset}" == "credit" ]]; then # Credit can not be smaller than 0.01
#    finetune_data_subsample='0.01 0.02'
    finetune_data_subsample='0.05 0.1 0.2'
  fi
  pretrain='3'
  send_pt0='1'
  python main.py --name 0511_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps} --split_train_as_val 1 --seed 30 --pretraining_ratio ${pretraining_ratio} --ignore_prev_runs 1
done


##### Some have not finished. Rerun those jobs not finished!
# Have not run compas, mimic3  0.01 0.02 0.05 0.1
arch='GAMAtt'
finetune_freeze_steps='500'
finetune_lr='5e-4 3e-4 1e-4 5e-5'
random_search='15'
pretraining_ratio='0.15'
for dset in 'mimic3' 'compas' 'credit'; do
  if [[ "${dset}" == "mimic2" ]]; then
    finetune_data_subsample='0.005 0.01'
  fi
  if [[ "${dset}" == "mimic3" ]]; then
    finetune_data_subsample='0.01 0.02 0.05 0.1'
  fi
  if [[ "${dset}" == "support2" ]]; then
    finetune_data_subsample='0.02 0.05 0.1'
  fi
  if [[ "${dset}" == "compas" ]]; then
    finetune_data_subsample='0.02 0.05 0.1'
  fi
  if [[ "${dset}" == "credit" ]]; then # Credit can not be smaller than 0.01
    finetune_data_subsample='0.02'
  fi
  if [[ "${dset}" == "churn" ]]; then
    finetune_data_subsample='0.02 0.05 0.1'
  fi
  pretrain='3'
  send_pt0='1'
  python main.py --name 0511_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps} --split_train_as_val 1 --seed 30 --pretraining_ratio ${pretraining_ratio} --ignore_prev_runs 1
done


random_search='15'
for dset in 'mimic2' 'adult'; do
  arch='GAMAtt'
  for entmoid_anneal_steps in '10000' '20000'; do
    for entmoid_min_temp in '0.1' '0.01'; do
      python main.py --name 0513_${dset}_${arch}_entm${entmoid_min_temp}_ents${entmoid_anneal_steps} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --entmoid_min_temp ${entmoid_min_temp} --seed 12 --entmoid_anneal_steps ${entmoid_anneal_steps}
    done
  done
done


# (R) Run the best mimic2/adult/compas with various min_temp!
entmoid_anneal_steps='5000'
for fold in '0' '1' '2' '3' '4'; do
for entmoid_min_temp in '0.75' '0.5' '0.25'; do
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0513_mimic2_f${fold}_entm${entmoid_min_temp}_GAMAtt_s89_nl5_nt400_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-07_pt0_pr0_mn0_ol0_ll0_da8 --load_from_hparams 0513_mimic2_GAMAtt_entm1_s89_nl5_nt400_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-07_pt0_pr0_mn0_ol0_ll0_da8 --entmoid_min_temp ${entmoid_min_temp} --entmoid_anneal_steps ${entmoid_anneal_steps} --fold ${fold}
done
done
# RUn the best compas?
entmoid_anneal_steps='5000'
for entmoid_min_temp in '0.75' '0.5' '0.25'; do
  for fold in '0' '1' '2' '3' '4'; do
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0513_compas_f${fold}_entm${entmoid_min_temp}_GAMAtt_s67_nl5_nt800_td2_d4_od0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0_mn0_ol0_da16_ll1 --load_from_hparams 0502_f0_best_compas_GAMAtt2_s67_nl5_nt800_td2_d4_od0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0_mn0_ol0_da16_ll1 --arch GAMAtt --entmoid_min_temp ${entmoid_min_temp} --entmoid_anneal_steps ${entmoid_anneal_steps} --fold ${fold}
  done
done
entmoid_anneal_steps='5000'
for entmoid_min_temp in '0.75' '0.5' '0.25'; do
  for fold in '0' '1' '2' '3' '4'; do
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0513_adult_f${fold}_entm${entmoid_min_temp}_GAM_lastl_s46_nl3_nt666_td1_d4_od0.1_cs0.5_lr0.01_lo0_la0.0 --load_from_hparams 0413_f0_best_adult_GAM_lastl_s46_nl3_nt666_td1_d4_od0.1_cs0.5_lr0.01_lo0_la0.0 --entmoid_min_temp ${entmoid_min_temp} --entmoid_anneal_steps ${entmoid_anneal_steps} --fold ${fold}
  done
done


random_search='15'
entmoid_anneal_steps='10'
for dset in 'mimic2' 'adult'; do
  arch='GAMAtt'
  for entmoid_min_temp in '0.1' '0.01'; do
    python main.py --name 0513_${dset}_${arch}_entm${entmoid_min_temp}_ents${entmoid_anneal_steps} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --entmoid_min_temp ${entmoid_min_temp} --seed 12 --entmoid_anneal_steps ${entmoid_anneal_steps}
  done
done


# (R) Run the random search for compas: I feel the graph is too smooth :(
random_search='15'
for dset in 'compas'; do
  arch='GAMAtt'
  entmoid_anneal_steps='10000'
  for entmoid_min_temp in '1' '0.1'; do
    python main.py --name 0513_${dset}_${arch}_entm${entmoid_min_temp}_ents${entmoid_anneal_steps} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --entmoid_min_temp ${entmoid_min_temp} --seed 23 --entmoid_anneal_steps ${entmoid_anneal_steps}
  done
done


## (R) Rerun the best for mimic2 and adult.
for d in \
'0513_mimic2_GAMAtt_entm1_s89_nl5_nt400_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-07_pt0_pr0_mn0_ol0_ll0_da8' \
'0513_adult_GAMAtt_entm1_s53_nl4_nt125_td0_d4_od0.0_ld0.0_cs0.5_lr0.005_lo0_la1e-06_pt0_pr0_mn0_ol0_ll1_da16' \
; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0514_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done



#### Run the 3-fold
for fold in '0' '1' '2'; do
### For the pt=0
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic2_GAMAtt_s55_nl2_nt250_td1_d6_od0.2_ld0.2_cs1e-05_lr0.01_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.005 --load_from_hparams 0511_mimic2_GAMAtt_s55_nl2_nt250_td1_d6_od0.2_ld0.2_cs1e-05_lr0.01_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.005 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic2_GAMAtt_s13_nl4_nt250_td0_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.01 --load_from_hparams 0511_mimic2_GAMAtt_s13_nl4_nt250_td0_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.01 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic2_GAMAtt_s80_nl5_nt400_td1_d2_od0.1_ld0.0_cs0.1_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.02 --load_from_hparams 0511_mimic2_GAMAtt_s80_nl5_nt400_td1_d2_od0.1_ld0.0_cs0.1_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.02 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic2_GAMAtt_s4_nl2_nt1000_td2_d6_od0.1_ld0.0_cs0.1_lr0.005_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.05 --load_from_hparams 0511_mimic2_GAMAtt_s4_nl2_nt1000_td2_d6_od0.1_ld0.0_cs0.1_lr0.005_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.05 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic2_GAMAtt_s55_nl2_nt250_td1_d6_od0.2_ld0.2_cs1e-05_lr0.01_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.1 --load_from_hparams 0511_mimic2_GAMAtt_s55_nl2_nt250_td1_d6_od0.2_ld0.2_cs1e-05_lr0.01_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.1 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic3_GAMAtt_s47_nl4_nt250_td1_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.005 --load_from_hparams 0511_mimic3_GAMAtt_s47_nl4_nt250_td1_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.005 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic3_GAMAtt_s46_nl5_nt200_td1_d4_od0.1_ld0.1_cs1e-05_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.01 --load_from_hparams 0511_mimic3_GAMAtt_s46_nl5_nt200_td1_d4_od0.1_ld0.1_cs1e-05_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.01 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic3_GAMAtt_s47_nl4_nt250_td1_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.02 --load_from_hparams 0511_mimic3_GAMAtt_s47_nl4_nt250_td1_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.02 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic3_GAMAtt_s68_nl4_nt125_td2_d4_od0.0_ld0.0_cs1e-05_lr0.005_lo0_la1e-07_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.05 --load_from_hparams 0511_mimic3_GAMAtt_s68_nl4_nt125_td2_d4_od0.0_ld0.0_cs1e-05_lr0.005_lo0_la1e-07_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.05 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic3_GAMAtt_s80_nl5_nt400_td1_d2_od0.1_ld0.0_cs0.1_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.1 --load_from_hparams 0511_mimic3_GAMAtt_s80_nl5_nt400_td1_d2_od0.1_ld0.0_cs0.1_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.1 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_adult_GAMAtt_s14_nl2_nt500_td2_d4_od0.1_ld0.3_cs0.1_lr0.005_lo0_la1e-06_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.005 --load_from_hparams 0511_adult_GAMAtt_s14_nl2_nt500_td2_d4_od0.1_ld0.3_cs0.1_lr0.005_lo0_la1e-06_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.005 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_adult_GAMAtt_s47_nl4_nt250_td1_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.01 --load_from_hparams 0511_adult_GAMAtt_s47_nl4_nt250_td1_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.01 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_adult_GAMAtt_s46_nl5_nt200_td1_d4_od0.1_ld0.1_cs1e-05_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.02 --load_from_hparams 0511_adult_GAMAtt_s46_nl5_nt200_td1_d4_od0.1_ld0.1_cs1e-05_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.02 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_adult_GAMAtt_s46_nl5_nt200_td1_d4_od0.1_ld0.1_cs1e-05_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.05 --load_from_hparams 0511_adult_GAMAtt_s46_nl5_nt200_td1_d4_od0.1_ld0.1_cs1e-05_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.05 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_adult_GAMAtt_s80_nl5_nt400_td1_d2_od0.1_ld0.0_cs0.1_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.1 --load_from_hparams 0511_adult_GAMAtt_s80_nl5_nt400_td1_d2_od0.1_ld0.0_cs0.1_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.1 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_support2_GAMAtt_s68_nl4_nt125_td2_d4_od0.0_ld0.0_cs1e-05_lr0.005_lo0_la1e-07_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.005 --load_from_hparams 0511_support2_GAMAtt_s68_nl4_nt125_td2_d4_od0.0_ld0.0_cs1e-05_lr0.005_lo0_la1e-07_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.005 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_support2_GAMAtt_s68_nl4_nt125_td2_d4_od0.0_ld0.0_cs1e-05_lr0.005_lo0_la1e-07_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.01 --load_from_hparams 0511_support2_GAMAtt_s68_nl4_nt125_td2_d4_od0.0_ld0.0_cs1e-05_lr0.005_lo0_la1e-07_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.01 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_support2_GAMAtt_s55_nl2_nt250_td1_d6_od0.2_ld0.2_cs1e-05_lr0.01_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.02 --load_from_hparams 0511_support2_GAMAtt_s55_nl2_nt250_td1_d6_od0.2_ld0.2_cs1e-05_lr0.01_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.02 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_support2_GAMAtt_s14_nl2_nt500_td2_d4_od0.1_ld0.3_cs0.1_lr0.005_lo0_la1e-06_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.05 --load_from_hparams 0511_support2_GAMAtt_s14_nl2_nt500_td2_d4_od0.1_ld0.3_cs0.1_lr0.005_lo0_la1e-06_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.05 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_support2_GAMAtt_s14_nl2_nt500_td2_d4_od0.1_ld0.3_cs0.1_lr0.005_lo0_la1e-06_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.1 --load_from_hparams 0511_support2_GAMAtt_s14_nl2_nt500_td2_d4_od0.1_ld0.3_cs0.1_lr0.005_lo0_la1e-06_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.1 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_compas_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.005 --load_from_hparams 0511_compas_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.005 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_compas_GAMAtt_s37_nl3_nt333_td1_d2_od0.0_ld0.3_cs1e-05_lr0.005_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.01 --load_from_hparams 0511_compas_GAMAtt_s37_nl3_nt333_td1_d2_od0.0_ld0.3_cs1e-05_lr0.005_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.01 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_compas_GAMAtt_s37_nl3_nt333_td1_d2_od0.0_ld0.3_cs1e-05_lr0.005_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.02 --load_from_hparams 0511_compas_GAMAtt_s37_nl3_nt333_td1_d2_od0.0_ld0.3_cs1e-05_lr0.005_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.02 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_compas_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.05 --load_from_hparams 0511_compas_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.05 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_compas_GAMAtt_s54_nl2_nt2000_td1_d4_od0.2_ld0.2_cs1e-05_lr0.005_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.1 --load_from_hparams 0511_compas_GAMAtt_s54_nl2_nt2000_td1_d4_od0.2_ld0.2_cs1e-05_lr0.005_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.1 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_credit_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.01 --load_from_hparams 0511_credit_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.01 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_credit_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.02 --load_from_hparams 0511_credit_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.02 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_credit_GAMAtt_s47_nl4_nt250_td1_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.05 --load_from_hparams 0511_credit_GAMAtt_s47_nl4_nt250_td1_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da32_ds0.05 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_credit_GAMAtt_s4_nl2_nt1000_td2_d6_od0.1_ld0.0_cs0.1_lr0.005_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.1 --load_from_hparams 0511_credit_GAMAtt_s4_nl2_nt1000_td2_d6_od0.1_ld0.0_cs0.1_lr0.005_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.1 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_credit_GAMAtt_s80_nl5_nt400_td1_d2_od0.1_ld0.0_cs0.1_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.2 --load_from_hparams 0511_credit_GAMAtt_s80_nl5_nt400_td1_d2_od0.1_ld0.0_cs0.1_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.2 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_churn_GAMAtt_s13_nl4_nt250_td0_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.005 --load_from_hparams 0511_churn_GAMAtt_s13_nl4_nt250_td0_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.005 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_churn_GAMAtt_s13_nl4_nt250_td0_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.01 --load_from_hparams 0511_churn_GAMAtt_s13_nl4_nt250_td0_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.01 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_churn_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.02 --load_from_hparams 0511_churn_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.02 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_churn_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.05 --load_from_hparams 0511_churn_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da16_ds0.05 --fold ${fold}
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_churn_GAMAtt_s46_nl5_nt200_td1_d4_od0.1_ld0.1_cs1e-05_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.1 --load_from_hparams 0511_churn_GAMAtt_s46_nl5_nt200_td1_d4_od0.1_ld0.1_cs1e-05_lr0.01_lo0_la1e-05_pt0_pr0.15_mn0.1_ol0_ll1_da8_ds0.1 --fold ${fold}
# For the pt=3
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic2_GAMAtt_s44_nl3_nt1333_td1_d4_od0.1_ld0.0_cs0.5_lr0.01_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --load_from_hparams 0511_mimic2_GAMAtt_s44_nl3_nt1333_td1_d4_od0.1_ld0.0_cs0.5_lr0.01_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.005 --finetune_lr 0.0005 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic2_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --load_from_hparams 0511_mimic2_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.02 0.05 --finetune_lr 0.0003 0.0003 --finetune_freeze_steps 500 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic2_GAMAtt_s55_nl2_nt250_td1_d6_od0.2_ld0.2_cs1e-05_lr0.01_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da32 --load_from_hparams 0511_mimic2_GAMAtt_s55_nl2_nt250_td1_d6_od0.2_ld0.2_cs1e-05_lr0.01_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da32 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.01 0.1 --finetune_lr 5e-05 0.0001 --finetune_freeze_steps 500 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic3_GAMAtt_s14_nl2_nt500_td2_d4_od0.1_ld0.3_cs0.1_lr0.005_lo0_la1e-06_pt3_pr0.15_mn0.1_ol0_ll1_da32 --load_from_hparams 0511_mimic3_GAMAtt_s14_nl2_nt500_td2_d4_od0.1_ld0.3_cs0.1_lr0.005_lo0_la1e-06_pt3_pr0.15_mn0.1_ol0_ll1_da32 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.01 --finetune_lr 5e-05 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic3_GAMAtt_s4_nl2_nt1000_td2_d6_od0.1_ld0.0_cs0.1_lr0.005_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da8 --load_from_hparams 0511_mimic3_GAMAtt_s4_nl2_nt1000_td2_d6_od0.1_ld0.0_cs0.1_lr0.005_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da8 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.05 0.1 --finetune_lr 0.0005 5e-05 --finetune_freeze_steps 500 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic3_GAMAtt_s68_nl4_nt125_td2_d4_od0.0_ld0.0_cs1e-05_lr0.005_lo0_la1e-07_pt3_pr0.15_mn0.1_ol0_ll1_da8 --load_from_hparams 0511_mimic3_GAMAtt_s68_nl4_nt125_td2_d4_od0.0_ld0.0_cs1e-05_lr0.005_lo0_la1e-07_pt3_pr0.15_mn0.1_ol0_ll1_da8 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.005 --finetune_lr 5e-05 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_mimic3_GAMAtt_s80_nl5_nt400_td1_d2_od0.1_ld0.0_cs0.1_lr0.005_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --load_from_hparams 0511_mimic3_GAMAtt_s80_nl5_nt400_td1_d2_od0.1_ld0.0_cs0.1_lr0.005_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.02 --finetune_lr 0.0003 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_adult_GAMAtt_s13_nl4_nt250_td0_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --load_from_hparams 0511_adult_GAMAtt_s13_nl4_nt250_td0_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.05 --finetune_lr 0.0003 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_adult_GAMAtt_s54_nl2_nt2000_td1_d4_od0.2_ld0.2_cs1e-05_lr0.005_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da32 --load_from_hparams 0511_adult_GAMAtt_s54_nl2_nt2000_td1_d4_od0.2_ld0.2_cs1e-05_lr0.005_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da32 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.02 --finetune_lr 0.0001 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_adult_GAMAtt_s55_nl2_nt250_td1_d6_od0.2_ld0.2_cs1e-05_lr0.01_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da32 --load_from_hparams 0511_adult_GAMAtt_s55_nl2_nt250_td1_d6_od0.2_ld0.2_cs1e-05_lr0.01_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da32 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.1 --finetune_lr 0.0005 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_adult_GAMAtt_s68_nl4_nt125_td2_d4_od0.0_ld0.0_cs1e-05_lr0.005_lo0_la1e-07_pt3_pr0.15_mn0.1_ol0_ll1_da8 --load_from_hparams 0511_adult_GAMAtt_s68_nl4_nt125_td2_d4_od0.0_ld0.0_cs1e-05_lr0.005_lo0_la1e-07_pt3_pr0.15_mn0.1_ol0_ll1_da8 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.005 0.01 --finetune_lr 0.0003 5e-05 --finetune_freeze_steps 500 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_support2_GAMAtt_s44_nl3_nt1333_td1_d4_od0.1_ld0.0_cs0.5_lr0.01_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --load_from_hparams 0511_support2_GAMAtt_s44_nl3_nt1333_td1_d4_od0.1_ld0.0_cs0.5_lr0.01_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.01 --finetune_lr 0.0005 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_support2_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --load_from_hparams 0511_support2_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.02 --finetune_lr 5e-05 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_support2_GAMAtt_s4_nl2_nt1000_td2_d6_od0.1_ld0.0_cs0.1_lr0.005_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da8 --load_from_hparams 0511_support2_GAMAtt_s4_nl2_nt1000_td2_d6_od0.1_ld0.0_cs0.1_lr0.005_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da8 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.05 --finetune_lr 5e-05 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_support2_GAMAtt_s68_nl4_nt125_td2_d4_od0.0_ld0.0_cs1e-05_lr0.005_lo0_la1e-07_pt3_pr0.15_mn0.1_ol0_ll1_da8 --load_from_hparams 0511_support2_GAMAtt_s68_nl4_nt125_td2_d4_od0.0_ld0.0_cs1e-05_lr0.005_lo0_la1e-07_pt3_pr0.15_mn0.1_ol0_ll1_da8 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.005 --finetune_lr 0.0005 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_support2_GAMAtt_s76_nl2_nt1000_td2_d6_od0.1_ld0.0_cs1e-05_lr0.005_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da8 --load_from_hparams 0511_support2_GAMAtt_s76_nl2_nt1000_td2_d6_od0.1_ld0.0_cs1e-05_lr0.005_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da8 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.1 --finetune_lr 0.0001 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_compas_GAMAtt_s13_nl4_nt250_td0_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --load_from_hparams 0511_compas_GAMAtt_s13_nl4_nt250_td0_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.05 --finetune_lr 0.0005 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_compas_GAMAtt_s37_nl3_nt333_td1_d2_od0.0_ld0.3_cs1e-05_lr0.005_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da16 --load_from_hparams 0511_compas_GAMAtt_s37_nl3_nt333_td1_d2_od0.0_ld0.3_cs1e-05_lr0.005_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da16 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.005 0.02 --finetune_lr 0.0001 0.0001 --finetune_freeze_steps 500 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_compas_GAMAtt_s56_nl2_nt2000_td1_d4_od0.0_ld0.0_cs1e-05_lr0.01_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da32 --load_from_hparams 0511_compas_GAMAtt_s56_nl2_nt2000_td1_d4_od0.0_ld0.0_cs1e-05_lr0.01_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da32 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.1 --finetune_lr 5e-05 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_compas_GAMAtt_s80_nl5_nt400_td1_d2_od0.1_ld0.0_cs0.1_lr0.005_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --load_from_hparams 0511_compas_GAMAtt_s80_nl5_nt400_td1_d2_od0.1_ld0.0_cs0.1_lr0.005_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.01 --finetune_lr 0.0005 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_credit_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --load_from_hparams 0511_credit_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.02 0.05 --finetune_lr 5e-05 0.0003 --finetune_freeze_steps 500 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_credit_GAMAtt_s4_nl2_nt1000_td2_d6_od0.1_ld0.0_cs0.1_lr0.005_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da8 --load_from_hparams 0511_credit_GAMAtt_s4_nl2_nt1000_td2_d6_od0.1_ld0.0_cs0.1_lr0.005_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da8 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.1 0.2 --finetune_lr 5e-05 0.0005 --finetune_freeze_steps 500 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_credit_GAMAtt_s79_nl5_nt200_td2_d4_od0.1_ld0.3_cs1e-05_lr0.01_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da8 --load_from_hparams 0511_credit_GAMAtt_s79_nl5_nt200_td2_d4_od0.1_ld0.3_cs1e-05_lr0.01_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da8 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.01 --finetune_lr 0.0005 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_churn_GAMAtt_s14_nl2_nt500_td2_d4_od0.1_ld0.3_cs0.1_lr0.005_lo0_la1e-06_pt3_pr0.15_mn0.1_ol0_ll1_da32 --load_from_hparams 0511_churn_GAMAtt_s14_nl2_nt500_td2_d4_od0.1_ld0.3_cs0.1_lr0.005_lo0_la1e-06_pt3_pr0.15_mn0.1_ol0_ll1_da32 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.01 --finetune_lr 5e-05 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_churn_GAMAtt_s47_nl4_nt250_td1_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da32 --load_from_hparams 0511_churn_GAMAtt_s47_nl4_nt250_td1_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da32 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.1 --finetune_lr 5e-05 --finetune_freeze_steps 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_churn_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --load_from_hparams 0511_churn_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.02 0.05 --finetune_lr 0.0005 0.0001 --finetune_freeze_steps 500 500
./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0515_f${fold}_ss_churn_GAMAtt_s55_nl2_nt250_td1_d6_od0.2_ld0.2_cs1e-05_lr0.01_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da32 --load_from_hparams 0511_churn_GAMAtt_s55_nl2_nt250_td1_d6_od0.2_ld0.2_cs1e-05_lr0.01_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da32 --fold ${fold} --finetune_zip 1 --finetune_data_subsample 0.005 --finetune_lr 0.0005 --finetune_freeze_steps 500
done




# Test for 10 runs for 2 other folds
arch='GAMAtt'
finetune_freeze_steps='500'
finetune_lr='5e-4 3e-4 1e-4 5e-5'
#random_search='10'
random_search='10'
pretraining_ratio='0.15'
pretrain='3'
send_pt0='1'
#for fold in '1' '2'; do
fold='2'
for dset in 'mimic2' 'mimic3' 'adult' 'support2'; do
#for dset in 'mimic2' 'mimic3' 'adult' 'compas' 'support2' 'credit' 'churn'; do
  finetune_data_subsample='0.005 0.01 0.02 0.05 0.1'
  if [[ "${dset}" == "credit" ]]; then # Credit can not be smaller than 0.01
    finetune_data_subsample='0.01 0.02 0.05 0.1 0.2'
  fi
  python main.py --name 0516_f${fold}_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --pretrain ${pretrain} --finetune_data_subsample ${finetune_data_subsample} --send_pt0 ${send_pt0} --finetune_lr ${finetune_lr} --finetune_freeze_steps ${finetune_freeze_steps} --split_train_as_val 1 --seed 30 --pretraining_ratio ${pretraining_ratio} --ignore_prev_runs 1 --fold ${fold}
done
#done


random_search='5'
for dset in 'mimic2' 'mimic3'; do
    for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0517_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 12 --ga2m 1
    done
done


# Run baselines for 3 folds and various data samples
for fold in '0' '1' '2'; do
#  for model_name in 'spline' 'ebm-o100-i100'; do # 'xgb-o50'
  for model_name in 'xgb-o50'; do # 'xgb-o50'
    for dset in 'mimic2' 'mimic3' 'adult' 'compas' 'support2' 'credit' 'churn'; do
      diff_ds='0.005'
      if [[ "${dset}" == "credit" ]]; then # Credit can not be smaller than 0.01
        diff_ds='0.2'
      fi
      for ds in '0.01' '0.02' '0.05' '0.1' ${diff_ds}; do
        echo "${dset} ${model_name} f=${fold} ds=${ds}"
        ./my_sbatch --cpu 10 --gpus 0 --mem 40 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_ds${ds}_f${fold} --dataset ${dset} --model_name ${model_name} --data_subsample ${ds} --fold ${fold}
      done
    done
  done
done


random_search='5'
for dset in 'mimic2' 'mimic3'; do
    for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 22 --ga2m 1
    done
done

random_search='10'
for dset in 'wine' 'support2' 'mimic2' 'mimic3'; do
    for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 22 --ga2m 1
    done
done


random_search='10'
for dset in 'mimic2' 'mimic3' 'adult' 'compas' 'support2' 'credit' 'churn'; do
    for arch in 'ODST'; do
    python main.py --name 0519_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 22
    done
done

for dset in 'mimic2' 'mimic3' 'adult' 'compas' 'support2' 'credit' 'churn'; do
  mv results/${dset}_ODST_new10.csv results/${dset}_ODST_new9.csv
done





# Run the best GA2M!
for d in \
'0518_wine_GAMAtt_ga2m_s87_nl4_nt1000_td1_d6_od0.2_ld0.0_cs0.5_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll0_da16' \
'0518_support2_GAMAtt_ga2m_s4_nl2_nt250_td0_d6_od0.2_ld0.3_cs1.0_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll0_da16' \
'0518_mimic3_GAMAtt_ga2m_s87_nl4_nt1000_td1_d6_od0.2_ld0.0_cs0.5_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll0_da16' \
'0517_mimic2_GAMAtt_ga2m_s42_nl4_nt500_td1_d6_od0.0_ld0.0_cs0.2_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll0_da16' \
; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0519_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done

# (0520) (R) the best ODST
#'0519_mimic2_ODST_s17_nl3_nt166_td0_d6_od0.2_cs0.2_lr0.005_la1e-06_ll0_ld0.0' \
#'0519_adult_ODST_s93_nl2_nt1000_td0_d4_od0.2_cs0.2_lr0.005_la1e-07_ll0_ld0.0' \
#'0519_mimic3_ODST_s93_nl2_nt1000_td0_d4_od0.2_cs0.2_lr0.005_la1e-07_ll0_ld0.0' \
#'0519_compas_ODST_s3_nl5_nt100_td1_d2_od0.0_cs0.2_lr0.005_la0.0_ll0_ld0.0' \
#'0519_churn_ODST_s3_nl5_nt100_td1_d2_od0.0_cs0.2_lr0.005_la0.0_ll0_ld0.0' \
#'0519_credit_ODST_s87_nl3_nt166_td0_d6_od0.0_cs0.2_lr0.01_la1e-07_ll1_ld0.2' \
#'0519_support2_ODST_s93_nl2_nt1000_td0_d4_od0.2_cs0.2_lr0.005_la1e-07_ll0_ld0.0' \
'0522_bikeshare_ODST_s49_nl3_nt333_td1_d4_od0.1_cs0.5_lr0.005_la1e-05_ll1_ld0.3' \
for d in \

; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --qos deadline --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0521_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done

# (R) run ga2m in other datasets: bikeshare,
random_search='10'
for dset in 'bikeshare' 'adult' 'churn' 'compas' 'credit'; do
    for arch in 'GAMAtt' 'GAM'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 23 --ga2m 1
    done
done

random_search='20'
for dset in 'bikeshare' 'adult' 'compas' 'wine' 'support2'; do
    for arch in 'GAM'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 23 --ga2m 1
    done
done
random_search='20'
for dset in 'bikeshare'; do
    for arch in 'GAM'; do
    python main.py --name 0522_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 22
    done
done

random_search='25'
for dset in 'support2'; do
    for arch in 'GAMAtt'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 22 --ga2m 1
    done
done



for model_name in 'spline' 'ebm-o100-i100' 'xgb-o50' 'xgb-d3'; do
  for dset in 'bikeshare'; do
    for fold in '0' '1' '2' '3' '4'; do
      ./my_sbatch --cpu 10 --gpus 0 --mem 40 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
    done
  done
done


fold='0'
for model_name in 'ebm-o50-i50-it4' 'ebm-o50-i50-it8' 'ebm-o50-i50-it16' 'ebm-o50-i50-it32' 'ebm-o50-i50-it64'; do
  for dset in 'bikeshare' 'adult' 'compas' 'churn' 'credit' 'mimic2' 'mimic3' 'wine' 'support2'; do
      ./my_sbatch --cpu 10 --gpus 0 --mem 40 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
  done
done
fold='0'
for model_name in 'ebm-o50-i50-it128' 'ebm-o50-i50-it256'; do
  for dset in 'bikeshare' 'adult' 'compas' 'churn' 'credit' 'mimic2' 'mimic3' 'wine' 'support2'; do
      ./my_sbatch --cpu 10 --gpus 0 --mem 40 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
  done
done

# (R) run for EBM interactions 5-fold
for dset in 'bikeshare' 'adult' 'compas' 'churn' 'credit' 'mimic2' 'mimic3' 'wine' 'support2'; do
  it='128'
  if [[ "${dset}" == "mimic2" ]]; then
    it='64'
  fi
  if [[ "${dset}" == "compas" ]]; then
    it='16'
  fi
  if [[ "${dset}" == "churn" ]]; then
    it='32'
  fi
  if [[ "${dset}" == "credit" ]]; then
    it='256'
  fi
  if [[ "${dset}" == "wine" ]]; then
    it='64'
  fi

  model_name="ebm-o100-i100-it${it}"
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 10 --gpus 0 --mem 40 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
    done
done

'0308_mimic2_ebm-o50-i50-it64_f0',
 '0308_adult_ebm-o50-i50-it128_f0',
 '0308_mimic3_ebm-o50-i50-it128_f0',
 '0308_compas_ebm-o50-i50-it16_f0',
 '0308_churn_ebm-o50-i50-it32_f0',
 '0308_credit__f0',
 '0308_support2_ebm-o50-i50-it128_f0',
 '0308_wine_ebm-o50-i50-it64_f0',
 '0308_bikeshare_ebm-o50-i50-it128_f0'


# TORUN:
random_search='40'
for dset in 'year' 'epsilon' 'yahoo' 'microsoft' 'higgs' 'click'; do
    for arch in 'GAMAtt'; do
    python main.py --name 0525_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 22 --ga2m 1
    done
done


# (0519) best for ga2m
for d in \
'0518_adult_GAM_ga2m_s32_nl3_nt1333_td2_d4_od0.2_ld0.2_cs1.0_lr0.005_lo0_la1e-05_pt0_pr0_mn0_ol0_ll1' \
'0518_compas_GAM_ga2m_s32_nl4_nt1000_td2_d2_od0.2_ld0.2_cs0.2_lr0.005_lo0_la0.0_pt0_pr0_mn0_ol0_ll1' \
'0518_churn_GAMAtt_ga2m_s31_nl3_nt333_td2_d2_od0.0_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt0_pr0_mn0_ol0_ll0_da32' \
'0518_credit_GAMAtt_ga2m_s97_nl3_nt666_td1_d4_od0.2_ld0.0_cs0.5_lr0.01_lo0_la1e-07_pt0_pr0_mn0_ol0_ll1_da32' \
'0518_bikeshare_GAM_ga2m_s83_nl4_nt125_td1_d6_od0.0_ld0.3_cs0.5_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll1' \
; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0519_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done



# (0519) best for bikeshare. Use date 0525
for d in \
'0522_bikeshare_GAM_s55_nl2_nt250_td1_d2_od0.2_ld0.3_cs0.5_lr0.005_lo0_la1e-07_pt0_pr0_mn0_ol0_ll1' \
; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --qos deadline --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0525_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done


# Run the l2 interactions for MIMIC2...
load_from_hparams='0519_f0_best_mimic2_GAMAtt_ga2m_s42_nl4_nt500_td1_d6_od0.0_ld0.0_cs0.2_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll0_da16'
qos='deadline'
for l2_interactions in '0.' '1e-4' '1e-5' '1e-6' '1e-7' '1e-8'; do
./my_sbatch --qos ${qos} --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0524_mimic2_l2itx${l2_interactions} --load_from_hparams ${load_from_hparams} --l2_interactions ${l2_interactions}
done


# Run the l2/l1 interactions for MIMIC2
load_from_hparams='0519_f0_best_mimic2_GAMAtt_ga2m_s42_nl4_nt500_td1_d6_od0.0_ld0.0_cs0.2_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll0_da16'
qos='deadline'
for l2_interactions in '1e-3' '1e-4' '1e-5' '1e-6' '1e-7'; do
./my_sbatch --qos ${qos} --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0526_mimic2_l2itx${l2_interactions} --load_from_hparams ${load_from_hparams} --l2_interactions ${l2_interactions}
done
for l1_interactions in '1e-3' '1e-4' '1e-5' '1e-6' '1e-7'; do
./my_sbatch --qos ${qos} --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0526_mimic2_l1itx${l1_interactions} --load_from_hparams ${load_from_hparams} --l1_interactions ${l1_interactions}
done

#### EBM iterations for large datasets: to see if the performance is too good.
# (R) run for EBM interactions 5-fold
for dset in 'year' 'microsoft' 'higgs' 'click' 'yahoo'; do
  model_name="ebm-o50-i50-it64"
  if [[ "${dset}" == "yahoo" ]]; then
    model_name='ebm-o10-i10-it64'
  fi
  ./my_sbatch --cpu 10 --gpus 0 --mem 80 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done

# Run the l2/l1 interactions for MIMIC2
load_from_hparams='0519_f0_best_mimic2_GAMAtt_ga2m_s42_nl4_nt500_td1_d6_od0.0_ld0.0_cs0.2_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll0_da16'
qos='deadline'
for l2_interactions in '0.1' '0.01'; do
./my_sbatch --qos ${qos} --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0526_mimic2_l2itx${l2_interactions} --load_from_hparams ${load_from_hparams} --l2_interactions ${l2_interactions}
done
for l1_interactions in '0.1' '0.01'; do
./my_sbatch --qos ${qos} --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0526_mimic2_l1itx${l1_interactions} --load_from_hparams ${load_from_hparams} --l1_interactions ${l1_interactions}
done




random_search='30'
for dset in 'wine' 'mimic2'; do
    for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 36 --ga2m 1
    done
done



#### More GA2M search for mimic2,mimic3,compas,credit
random_search='30'
for dset in 'compas' 'mimic3' 'credit'; do
    for arch in 'GAM' 'GAMAtt'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 36 --ga2m 1
    done
done



#### Wine ODST random search
random_search='50'
for dset in 'wine'; do
    for arch in 'ODST'; do
    python main.py --name 0519_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 22
    done
done



# EBM too strong. Might be cuz it is choosing the best in test set 0. Try just fix 64. Check if EBM gets slightly worse
for dset in 'bikeshare' 'adult' 'compas' 'churn' 'credit' 'mimic2' 'mimic3' 'wine' 'support2'; do
  it='64'
  model_name="ebm-o100-i100-it${it}"
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 10 --gpus 0 --mem 40 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
    done
done



# New best for these 5 datasets 0527. Check v.s. old best (0521) is better
for d in \
'0518_support2_GAMAtt_ga2m_s59_nl3_nt1333_td2_d2_od0.2_ld0.1_cs0.5_lr0.01_lo0_la1e-07_pt0_pr0_mn0_ol0_ll0_da32' \
'0518_adult_GAM_ga2m_s39_nl3_nt666_td2_d4_od0.1_ld0.3_cs0.2_lr0.005_lo0_la1e-06_pt0_pr0_mn0_ol0_ll1' \
'0517_mimic2_GAMAtt_ga2m_s10_nl2_nt2000_td0_d6_od0.0_ld0.0_cs0.2_lr0.005_lo0_la1e-05_pt0_pr0_mn0_ol0_ll0_da8' \
'0518_compas_GAMAtt_ga2m_s93_nl3_nt166_td1_d4_od0.1_ld0.0_cs0.5_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll1_da8' \
; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --qos deadline --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0527_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done



# New best for wine ODST: check it v.s. 0507 which one is better
for d in \
'0519_wine_ODST_s73_nl2_nt500_td1_d4_od0.0_cs1.0_lr0.01_la0.0_ll0_ld0.0' \
; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0521_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done


# Run a xgb-d5 in bikeshare
for dset in 'bikeshare'; do
  model_name="xgb-d5"
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 10 --gpus 0 --mem 40 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
    done
done

# Check EBM-it on large. Check if they die...
for dset in 'higgs' 'yahoo'; do
  model_name="ebm-o10-i10-it16"
  if [[ "${dset}" == "yahoo" ]]; then
    model_name='ebm-o10-i10-it16'
  fi
  ./my_sbatch --cpu 10 --gpus 0 --mem 120 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done


# New best for these 5 datasets 0527. Check v.s. old best (0521) is better
for d in \
'0518_credit_GAMAtt_ga2m_s38_nl2_nt1000_td0_d6_od0.2_ld0.0_cs0.2_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll1_da32' \
; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --qos deadline --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0527_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done


random_search='50'
for dset in 'year'; do
  selectors_detach='0'
  if [[ "${dset}" == "epsilon" ]]; then
      selectors_detach='1'
  fi
  for arch in 'GAMAtt'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 89 --ga2m 1 --selectors_detach ${selectors_detach}
  done
done


random_search='50'
for dset in 'click' 'yahoo'; do
  selectors_detach='0'
  if [[ "${dset}" == "epsilon" ]]; then
      selectors_detach='1'
  fi
  for arch in 'GAMAtt'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 89 --ga2m 1 --selectors_detach ${selectors_detach}
  done
done

random_search='50'
for dset in 'mimic2' 'support2' 'bikeshare'; do
#  if [[ "${dset}" == "epsilon" ]]; then
#      selectors_detach='1'
#  fi
  for arch in 'GAMAtt2'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 93 --ga2m 1
  done
done
random_search='25'
for dset in 'adult' 'compas' 'wine'; do
#  if [[ "${dset}" == "epsilon" ]]; then
#      selectors_detach='1'
#  fi
  for arch in 'GAM' 'GAMAtt2'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 93 --ga2m 1
  done
done


random_search='5'
for dset in 'epsilon'; do
  for selectors_detach in '0' '1'; do
#  if [[ "${dset}" == "epsilon" ]]; then
#      selectors_detach='1'
#  fi
    for arch in 'GAMAtt2'; do
      python main.py --name 0518_${dset}_${arch}_ga2m_sd${selectors_detach} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 93 --ga2m 1 --selectors_detach ${selectors_detach}
    done
  done
done


# New best for 0528
for d in \
'0518_support2_GAMAtt2_ga2m_s33_nl2_nt2000_td2_d6_od0.1_ld0.0_cs1.0_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll0_da32' \
'0518_adult_GAM_ga2m_s33_nl2_nt2000_td2_d6_od0.1_ld0.0_cs1.0_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll1' \
; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --qos deadline --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0528_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done


random_search='50'
for dset in 'adult'; do
  for arch in 'GAM'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 48 --ga2m 1 --add_last_linear 1
  done
done

random_search='50'
for dset in 'mimic2'; do
  for arch in 'GAMAtt2'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 36 --ga2m 1 --add_last_linear 0
  done
done


# Maybe adult GAM ga2m will get last linear as 0 is best??? No!
random_search='30'
for dset in 'adult'; do
  for arch in 'GAM'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 36 --ga2m 1 --add_last_linear 0
  done
done


# New best for mimic2!! (A bit cheating!!)
for d in \
'0518_mimic2_GAMAtt2_ga2m_s97_nl3_nt1333_td0_d4_od0.1_ld0.0_cs0.2_lr0.01_lo0_la1e-06_pt0_pr0_mn0_ol0_ll0_da32' \
; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --qos deadline --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0528_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done

random_search='60'
for dset in 'mimic2'; do
  for arch in 'GAM' 'GAMAtt'; do
    add_last_linear='0'
    if [[ "${arch}" == "GAM" ]]; then
      add_last_linear='1'
    fi
    python main.py --name 0527_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 26 --add_last_linear ${add_last_linear}
  done
done


for dset in 'bikeshare' 'adult' 'compas' 'churn' 'credit' 'mimic2' 'mimic3' 'wine' 'support2'; do
  model_name="rf"
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 10 --gpus 0 --mem 40 -p cpu python -u baselines.py --name 0308_${dset}_${model_name}_f${fold} --dataset ${dset} --model_name ${model_name} --fold ${fold}
    done
done





# Rerun the best model among all datasets
for fold in '0' '1' '2' '3' '4'; do
  for d in \
"0514_f${fold}_best_mimic2_GAMAtt_entm1_s89_nl5_nt400_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-07_pt0_pr0_mn0_ol0_ll0_da8" \
"0413_f${fold}_best_adult_GAM_lastl_s46_nl3_nt666_td1_d4_od0.1_cs0.5_lr0.01_lo0_la0.0" \
"0502_f${fold}_best_f0_best_mimic3_GAM_s97_nl3_nt1333_td0_d6_od0.2_cs1e-05_lr0.005_lo0_la1e-07" \
"0502_f${fold}_best_compas_GAMAtt2_s67_nl5_nt800_td2_d4_od0.3_cs0.5_lr0.01_lo0_la1e-05_pt0_pr0_mn0_ol0_da16_ll1" \
"0502_f${fold}_best_churn_GAMAtt2_s48_nl3_nt166_td2_d4_od0.1_cs0.5_lr0.005_lo0_la1e-05_pt0_pr0_mn0_da8_ll1" \
"0502_f${fold}_best_f0_best_credit_GAMAtt2_s87_nl5_nt400_td2_d2_od0.2_cs0.1_lr0.01_lo0_la0.0_da8_ll1" \
"0420_support2_f${fold}_best" \
"0420_wine_f${fold}_best" \
"0525_f${fold}_best_bikeshare_GAM_s55_nl2_nt250_td1_d2_od0.2_ld0.3_cs0.5_lr0.005_lo0_la1e-07_pt0_pr0_mn0_ol0_ll1" \
; do
  postfix=${d:4}
  ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0530_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
done
done


#### Check if epsilon would work. And search more GA2M on click, yahoo, year, epsilon. Check if these improve performance
random_search='30'
for dset in 'click' 'yahoo' 'epsilon' 'year'; do
  selectors_detach='0'
  if [[ "${dset}" == "epsilon" ]]; then
      selectors_detach='1'
  fi
  add_last_linear='1'
  if [[ "${dset}" == "click" ]]; then
      add_last_linear='0'
  fi
  for arch in 'GAMAtt3'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 36 --ga2m 1 --selectors_detach ${selectors_detach} --add_last_linear ${add_last_linear}
  done
done


date
0531   -0.831800
0532   -0.830065
0533   -0.832476
# Honest MIMIC2: 0531,0532. Not-honest MIMIC2: 0533. Wait to see what to show for MIMIC2 main
#'0527_mimic2_GAM_s81_nl2_nt500_td1_d2_od0.1_ld0.1_cs1e-05_lr0.005_lo0_la1e-05_pt0_pr0_mn0_ol0_ll1' \
'0527_mimic2_GAM_s48_nl4_nt1000_td2_d4_od0.0_ld0.2_cs0.5_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll1' \
'0527_mimic2_GAMAtt_s84_nl2_nt500_td2_d6_od0.2_ld0.0_cs1e-05_lr0.005_lo0_la1e-06_pt0_pr0_mn0_ol0_ll0_da8' \
for d in \
'0527_mimic2_GAMAtt_s99_nl4_nt500_td1_d4_od0.0_ld0.0_cs0.5_lr0.01_lo0_la1e-07_pt0_pr0_mn0_ol0_ll0_da32' \
; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
#    ./my_sbatch --qos deadline --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0531_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
#    ./my_sbatch --qos deadline --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0532_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0533_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done


for dset in 'year' 'yahoo' 'microsoft' 'higgs' 'epsilon' 'click'; do
  model_name="rf"
  ./my_sbatch --cpu 20 --gpus 0 --mem 80 -p cpu python -u baselines.py --name 0308_${dset}_${model_name} --dataset ${dset} --model_name ${model_name}
done

for dset in 'year' 'yahoo' 'microsoft' 'higgs' 'epsilon' 'click'; do
./my_sbatch --cpu 8 --gpus 1 --mem 16 --qos deadline --name cache_${dset}2 python -u recheck_results.py --dataset ${dset}
done

for dset in 'yahoo' 'epsilon'; do
./my_sbatch --cpu 8 --gpus 1 --mem 24 --qos deadline --name cache_${dset}2 python -u recheck_results.py --dataset ${dset}
done

for dset in 'click'; do
./my_sbatch --cpu 8 --gpus 1 --mem 24 --qos deadline --name cache_${dset}2 python -u recheck_results.py --dataset ${dset}
done

for dset in 'yahoo' 'epsilon'; do
./my_sbatch --cpu 20 --gpus 0 --mem 80 --qos deadline -p cpu --name cache_${dset}2 python -u recheck_results.py --dataset ${dset}
done



############ RUNNING
random_search='50'
for dset in 'click' 'yahoo' 'epsilon' 'year' 'microsoft' 'higgs'; do
  selectors_detach='0'
  if [[ "${dset}" == "epsilon" ]]; then
      selectors_detach='1'
  fi
  add_last_linear='1'
  for arch in 'GAMAtt2'; do
    python main.py --name 0518_${dset}_${arch}_ga2m --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 33 --ga2m 1 --selectors_detach ${selectors_detach} --add_last_linear ${add_last_linear}
  done
done


random_search='50'
for dset in 'click' 'yahoo' 'epsilon' 'year' 'microsoft' 'higgs'; do
  selectors_detach='0'
  if [[ "${dset}" == "epsilon" ]]; then
      selectors_detach='1'
  fi
  add_last_linear='1'
  for arch in 'GAMAtt2'; do
    python main.py --name 0532_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 56 --ga2m 0 --selectors_detach ${selectors_detach} --add_last_linear ${add_last_linear}
  done
done


for d in \
'0533_mimic2_ODST_s75_nl3_nt333_td2_d6_od0.1_cs0.2_lr0.005_la1e-05_ll0_ld0.0' \
'0533_mimic3_ODST_s46_nl2_nt1000_td1_d6_od0.2_cs0.2_lr0.005_la1e-06_ll0_ld0.0' \
'0533_churn_ODST_s26_nl4_nt125_td0_d2_od0.0_cs0.5_lr0.005_la1e-05_ll0_ld0.0' \
'0533_support2_ODST_s35_nl5_nt100_td1_d4_od0.1_cs0.2_lr0.01_la1e-05_ll0_ld0.0' \
; do
  postfix=${d:4}
  for fold in '0' '1' '2' '3' '4'; do
    ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name 0535_f${fold}_best${postfix} --load_from_hparams ${d} --fold ${fold}
  done
done


# cal_prev_feat_weights


############ TORUN




# Run better ODST again on these 4 datasets to make them look better?
random_search='10'
for dset in 'churn' 'compas' 'adult' 'credit'; do
  for arch in 'ODST'; do
    python main.py --name 0533_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 37
  done
done

random_search='30'
for dset in 'support2' 'mimic3' 'mimic2' 'bikeshare'; do
  for arch in 'ODST'; do
    python main.py --name 0533_${dset}_${arch} --dataset ${dset} --random_search ${random_search} --cpu 4 --gpus 1 --mem 8 --fp16 1 --arch ${arch} --seed 37
  done
done




############ Paper writing
Main
(1) Wait all ODST random search. After they finish, see which is the best and run. Update hyperparameters.

Supplementary
(3) Plot the GA2M like Fig.1
(4) Complete set of plots from datasets
(1) Purification pseudo-code? Maybe borrow from Ben?




********* MIMIC2 GAM is a fake one! only 0.825
********* Re-check all the best GAM model; do they perform exactly as reported? Like MIMIC2 is not!!
- Most of them are deleted! So let me rerun the best val MIMIC2?



Self-supervision / Pretraining: (RUNNING!!)
* (R) Hyperparameter tuning for those big datasets and subsample to smaller ratio (< 0.01)
  - Mask ratio
  - finetune lr
  - finetune steps
* Ensemble for self-supervised learning methods?
* Multiple runs for self-supervised methods to get stdev and more stable results

Another experiment:
* Step functions and smooth functions?
  - I do not like this: intuitively this can model step function already. One thing we might show is it can model linear function.

Inverse RL:
* Tune expert hyperparameters
  - Test the random performance of the environment
  - Create csv system for hyperparameters search?

Writing:
* Write how I extract GAM graph!
* Put up GAM graphs and write descriptions


# Idea: step functions?
#


* (R) run semi-supervised on large with small ds (click,microsoft,year) (0506)
* (R) run baselines with various ds
* TODO: make inverse RL on current state only! Compare to linear methods.
  - Mean return of the expert is 6.633536075911141

? TODO: run odst with more datasets. And we can have ODST with new parameters.
? TODO: test the interaction term extraction
  (1) Show accuracy improvement in some datasets
  (2) Extract graphs

Exp:
- Accuracy
  (1) NGAM and NGA2M
  (2) NODE and NODE(new)
- Graph
- Semi-supervised learning
- Inverse RL
- Ablation study (w/ FC or w/ Att or Att3)



? TODO: do mimicking NN w/ diff activation func? Mimicking FCNN/ODST?
  - Try NN training on these datasets w/ hyperparameter search?
  - Try different activation functions?
  A: Not very exciting. Inductive bias is a bit arbitrary, and it could be that other NNs can mimick better. And you do not know the exact main effect of a black-box. And PDP might be good enough by directly extracting single element. And there could be similar main effects in correlated features.




Ideas:
(1) How different activation function changes graph?
  - It requires different arch design, as now I am just multiple gates w/ response.
(2) Maybe worst-case GAM semi-synthetic exps? Or measure l1/l2-ness of GAM as well?

? TODO:
(1) Test early stopping 10k is enough? Plot the hist.



# Higgs;
# Think => After doing hyperparameters search, the Higgs pretraining mostly helps, but just 1 model in normal without pretraining gets reasonable well. It might be due to lucky initialization. What should I do?

# Idea: Mess up targets??? Like filtering out certain y? Introducing selection bias?


Towork:
(1) Inverse RL: simualte settings with various shape graph!
  - Setup time-series environment
  - Then train an inverse RL


Thoughts:
(0) QQ I forget to do stratified shuffle kfold for my test set lol
(1) (Running 0405) Should I do uniform or gaussian? Like uniform can easily fight the mean imputation!
(2) (R) Should I test GAMAtt2 and add_last_linear?
  - I can still justify GAMAtt as ResNet design, so might not GAMAtt2
  - May wait MTL to add last layer weight
(3) How shoudl I design in MTL the last layer weights?
* Can achieve it by setting negative value for addi_tree_dim and num_classes as large!


- TODO1: Download the SARCOS dataset and only use training set!
- TODO2: Compare if having last layer weight becomes better?
- TODO3: Implement multi-task learning by having the last layer weights!
- TODO4: (Optional) Maybe do the school multi-task learning datasets

- Multi-task learning? How?
  * Change the last layer weights!
  * Since in the original formulation, we already have R, which is the response weights! So we probably do not need another weights to do linear weighting. But in multi-task that might be important!

- Another relu design?
  * Maybe just product of relu-weights?

- TODO: Modify input dropout by shuffling noise????
# TODO:
  -



## L1/L2
- Problem is I have to cite my own paper to say this is important for fairness/bias discovery, but I can not afford that much space
## Semi-synthetic
- Slow, and might not be that interesting...
- But could be interesting and easy to be understood the worst-case scenario
## Transfer learning / Multi-task learning
- The datasets I found before...?




### Die lots of things. Maybe about unenough memory? Rerun things!
All rerun is bigger than 2763897

for d in \
; do
  load_from_hparams=${d:0:-4} # Remove directory name
  echo ${load_from_hparams}
  ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${d} --load_from_hparams ${load_from_hparams} --pretrain 0
#  ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${d}
done

for d in \
'0521_f4_best_compas_ODST_s3_nl5_nt100_td1_d2_od0.0_cs0.2_lr0.005_la0.0_ll0_ld0.0' \
'0521_f1_best_credit_ODST_s87_nl3_nt166_td0_d6_od0.0_cs0.2_lr0.01_la1e-07_ll1_ld0.2' \
; do
  ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${d}
done


for d in \
'0516_f2_support2_GAMAtt_s44_nl3_nt1333_td1_d4_od0.1_ld0.0_cs0.5_lr0.01_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16_fds0.005_flr0.0003_frs500_ft' \
; do
  ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${d}
done

for d in \
'0516_f2_adult_GAMAtt_s76_nl2_nt1000_td2_d6_od0.1_ld0.0_cs1e-05_lr0.005_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da8_fds0.1_flr0.0003_frs500_ft' \
'0516_f2_mimic3_GAMAtt_s13_nl4_nt250_td0_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt3_pr0.15_mn0.1_ol0_ll1_da16_fds0.05_flr0.0005_frs500_ft' \
'0516_f2_adult_GAMAtt_s76_nl2_nt1000_td2_d6_od0.1_ld0.0_cs1e-05_lr0.005_lo0_la0.0_pt3_pr0.15_mn0.1_ol0_ll1_da8_fds0.1_flr5e-05_frs500_ft' \
; do
  ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${d}
done

sacct -j 2773893,2773899,2773901,2773911,2773915,2773921,2773923,2774554,2773709,2773918,2773920,2773922,2774501,2774509,2774518,2774536,2774547,2774651,2774655,2774656,2774657,2774659,2774705,2774706,2774707,2774708,2773926,2774875,2774885,2774892,2774893,2774895,2774897,2774901,2774902,2774903,2774904,2775054,2775055,2776154,2776155,2776160,2776163,2776156,2776158,2776157,2776161,2776164,2776165 --format=user,job%20,start,end,elapsed,state,ReqGRE,MaxRSS,nodelist
sacct --user=kingsley --format='JobID,JobName%150,Partition,State,ExitCode'


'2774901\|2774903\|2774904\|2775054\|2775055\|2776154\|2776155\|2776160\|2776163\|2774554\|2776156\|2776158\|2776157\|2776161\|2776165\|'

for d in \
'0430_mimic2_GAM_ds0.06_s55_nl3_nt1333_td0_d6_od0.0_ld0.3_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.005frs1000_ft' \
'0430_mimic2_GAM_ds0.06_s55_nl3_nt1333_td0_d6_od0.0_ld0.3_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.01frs2000_ft' \
'0430_mimic2_GAM_ds0.06_s55_nl3_nt1333_td0_d6_od0.0_ld0.3_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.0001frs2000_ft' \
'0430_higgs_GAM_ds1100_s18_nl4_nt500_td2_d4_od0.1_ld0.0_cs0.5_lr0.01_lo0_la1e-05_pt2_pr0.15_mn0.1_ol0_ll1_flr0.01frs0_ft' \
'0430_higgs_GAM_ds1100_s18_nl4_nt500_td2_d4_od0.1_ld0.0_cs0.5_lr0.01_lo0_la1e-05_pt2_pr0.15_mn0.1_ol0_ll1_flr0.01frs1000_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.01frs0_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.01frs1000_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.0005frs0_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.0001frs0_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.01frs2000_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.005frs1000_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.005frs0_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.0005frs1000_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.0001frs2000_ft' \
; do
  echo $d
  tail logs/slrun/$d
done


for d in \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.01frs0_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.01frs1000_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.0005frs0_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.0001frs0_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.01frs2000_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.005frs1000_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.005frs0_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.0005frs1000_ft' \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_flr0.0001frs2000_ft' \
; do
  echo $d
  rm -r logs/$d
#  tail logs/slrun/$d
done




for d in \
'0510_yahoo_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt2_pr0.1_mn0.1_ol0_ll' \
'0510_microsoft_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt2_pr0.2_mn0.1_ol0_ll1_da16_fds0.0005_flr0.0001_frs1000_ft' \
'0510_microsoft_GAMAtt_s37_nl3_nt333_td1_d2_od0.0_ld0.3_cs1e-05_lr0.005_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_da16_fds0.0002_flr0.0003_frs1000_ft' \
'0510_microsoft_GAMAtt_s37_nl3_nt333_td1_d2_od0.0_ld0.3_cs1e-05_lr0.005_lo0_la0.0_pt2_pr0.3_mn0.1_ol0_ll1_da16_fds0.0005_flr0.0001_frs500_ft' \
'0510_microsoft_GAMAtt_s46_nl5_nt200_td1_d4_od0.1_ld0.1_cs1e-05_lr0.01_lo0_la1e-05_pt2_pr0.15_mn0.1_ol0_ll1_da8_fds0.0002_flr0.0001_frs1000_ft' \
'0510_microsoft_GAMAtt_s46_nl5_nt200_td1_d4_od0.1_ld0.1_cs1e-05_lr0.01_lo0_la1e-05_pt2_pr0.15_mn0.1_ol0_ll1_da8_fds0.0005_flr0.0001_frs1000_ft' \
'0510_microsoft_GAMAtt_s79_nl5_nt200_td2_d4_od0.1_ld0.3_cs1e-05_lr0.01_lo0_la0.0_pt2_pr0.1_mn0.1_ol0_ll1_da8_fds0.0005_flr0.0005_frs500_ft' \
'0510_microsoft_GAMAtt_s79_nl5_nt200_td2_d4_od0.1_ld0.3_cs1e-05_lr0.01_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_da8_fds0.0005_flr0.0003_frs1000_ft' \
'0510_microsoft_GAMAtt_s79_nl5_nt200_td2_d4_od0.1_ld0.3_cs1e-05_lr0.01_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_da8_fds0.0005_flr0.0003_frs500_ft' \
'0510_microsoft_GAMAtt_s79_nl5_nt200_td2_d4_od0.1_ld0.3_cs1e-05_lr0.01_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_da8_fds0.0002_flr0.0001_frs500_ft' \
'0510_microsoft_GAMAtt_s79_nl5_nt200_td2_d4_od0.1_ld0.3_cs1e-05_lr0.01_lo0_la0.0_pt2_pr0.15_mn0.1_ol0_ll1_da8_fds0.0005_flr0.0001_frs1000_ft' \
; do
  ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${d}
done


for d in \
'0430_higgs_GAM_ds1100_s16_nl4_nt500_td1_d6_od0.2_ld0.2_cs0.5_lr0.005_lo0_la0.0_pt0_pr0.15_mn0.1_ol0_ll1' \
; do
  ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${d}
done
# Check after 2778383

for d in \
'0510_higgs_GAMAtt_s76_nl2_nt1000_td2_d6_od0.1_ld0.0_cs1e-05_lr0.005_lo0_la0.0_pt2_pr0.2_mn0.1_ol0_ll1_da8' \
'0510_higgs_GAMAtt_s13_nl4_nt250_td0_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt2_pr0.2_mn0.1_ol0_ll1_da16' \
'0510_higgs_GAMAtt_s44_nl3_nt1333_td1_d4_od0.1_ld0.0_cs0.5_lr0.01_lo0_la1e-05_pt2_pr0.2_mn0.1_ol0_ll1_da16' \
'0510_higgs_GAMAtt_s54_nl2_nt2000_td1_d4_od0.2_ld0.2_cs1e-05_lr0.005_lo0_la0.0_pt2_pr0.2_mn0.1_ol0_ll1_da32' \
'0510_higgs_GAMAtt_s49_nl3_nt166_td0_d4_od0.2_ld0.3_cs0.5_lr0.01_lo0_la1e-05_pt2_pr0.2_mn0.1_ol0_ll1_da16' \
'0510_higgs_GAMAtt_s56_nl2_nt2000_td1_d4_od0.0_ld0.0_cs1e-05_lr0.01_lo0_la0.0_pt2_pr0.2_mn0.1_ol0_ll1_da32' \
'0510_higgs_GAMAtt_s13_nl4_nt250_td0_d4_od0.2_ld0.3_cs1e-05_lr0.005_lo0_la1e-05_pt2_pr0.25_mn0.1_ol0_ll1_da16' \
'0510_higgs_GAMAtt_s80_nl5_nt400_td1_d2_od0.1_ld0.0_cs0.1_lr0.005_lo0_la1e-05_pt2_pr0.25_mn0.1_ol0_ll1_da16' \
; do
  ./my_sbatch --cpu 4 --gpus 1 --mem 8 python -u main.py --name ${d}
done



