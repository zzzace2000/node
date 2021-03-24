
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




## TODO: split the main file into 3 different model architecture? GAM, GAMAtt, ODST.
# The ODST would be the original baseline
## TODO: change the learning rate? Search LR?

