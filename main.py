import os, sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import lib
import torch, torch.nn as nn
import torch.nn.functional as F
import argparse
import json
from os.path import join as pjoin, exists as pexists
import shutil
import platform
from filelock import FileLock
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
import subprocess
import os
import time


# Don't use multiple gpus; If more than 1 gpu, just use first one
if torch.cuda.device_count() > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Use it to create figure instead of interactive
matplotlib.use('Agg')

def get_rs_name(args, rs_hparams):
    return '_'.join(f'{v["short_name"]}{getattr(args, k)}'
                    for k, v in rs_hparams.items())

def get_args():
    # Big Transfer arg parser
    parser = argparse.ArgumentParser(description="Fine-tune BiT-M model.")
    parser.add_argument("--name",
                        default='debug',
                        help="Name of this run. Used for monitoring and checkpointing.")
    parser.add_argument('--seed', type=int, default=2,
                        help='seed for initializing training.')
    # My own arguments
    parser.add_argument("--dataset", default='mimic2',
                        help="Choose the dataset.",
                        choices=['year', 'epsilon', 'a9a', 'higgs', 'microsoft',
                                 'yahoo', 'click', 'mimic2', 'adult', 'churn',
                                 'credit', 'compas', 'support2', 'mimic3'])
    parser.add_argument('--fold', type=int, default=0,
                        help='Choose from 0 to 4, as we only support 5-fold CV.')
    parser.add_argument("--arch", type=str, default='GAM', choices=['ODST', 'GAM', 'GAMAtt'])
    parser.add_argument("--num_trees", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--addi_tree_dim", type=int, default=0)
    parser.add_argument("--l2_lambda", type=float, default=0.)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--lr_warmup_steps", type=int, default=-1)
    parser.add_argument("--lr_decay_steps", type=int, default=-1,
                        help='Decay learning rate by 1/5 if not improving for this step')

    parser.add_argument("--early_stopping_rounds", type=int, default=11000)
    parser.add_argument("--max_rounds", type=int, default=-1)
    parser.add_argument("--max_time", type=float, default=3600 * 20) # At most 20 hours
    parser.add_argument("--report_frequency", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_bs", type=int, default=2048)
    parser.add_argument("--min_bs", type=int, default=128)

    parser.add_argument("--random_search", type=int, default=0)
    parser.add_argument('--fp16', type=int, default=1,
                        help='Slows down 5~10%. But saves memory and is slightly better')

    # Only used in random search and when using slurm
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--mem', type=int, default=8)
    parser.add_argument('--cpu', type=int, default=4)

    temp_args, _ = parser.parse_known_args()
    if temp_args.name == 'debug':
        temp_args.arch = 'GAM'

    parser = getattr(lib.arch, temp_args.arch + 'Block').add_model_specific_args(parser)
    args = parser.parse_args()

    # Remove debug folder
    if args.name == 'debug':
        print("WATCHOUT!!! YOU ARE RUNNING IN TEST RUN MODE!!!")
        if pexists('./logs/debug'):
            shutil.rmtree('./logs/debug', ignore_errors=True)

        # args.arch = 'ODST'
        args.arch = 'GAM'
        # args.arch = 'GAM'
        # args.dim_att = 50
        # args.colsample_bytree = 1.0
        # args.output_dropout = 0.2
        # args.num_layers = 8
        args.num_layers = 2
        args.num_trees = 100
        args.addi_tree_dim = 1
        # args.num_trees = 16000 // 8
        # args.depth = 1
        # args.l2_lambda = 0.
        # args.lr = 0.02
        # args.batch_size = None
        args.batch_size = 1024
        # args.fp16 = 1
        args.depth = 2
        # args.anneal_steps = 100

    saved_hparams = pjoin('logs', args.name, 'hparams.json')
    if args.name != 'debug' and pexists(saved_hparams):
        hparams = json.load(open(saved_hparams))

        # Remove default value. Only parse user inputs
        for action in parser._actions:
            action.default = argparse.SUPPRESS
        input_args = parser.parse_args()
        print('Reload and update from inputs: ' + str(input_args))
        if len(vars(input_args)) > 0:
            hparams.update(vars(input_args))
            json.dump(hparams, open(saved_hparams, 'w'))
        args = argparse.Namespace(**hparams)

    # on v server
    if not pexists(pjoin('logs', args.name)) \
            and 'SLURM_JOB_ID' in os.environ \
            and pexists('/checkpoint/kingsley/%s' % os.environ['SLURM_JOB_ID']):
        os.symlink('/checkpoint/kingsley/%s' % os.environ['SLURM_JOB_ID'],
                   pjoin('logs', args.name))
    return args


def main(args) -> None:
    # Create directory
    os.makedirs(pjoin('logs', args.name), exist_ok=True)

    if pexists(pjoin('logs', args.name, 'MY_IS_FINISHED')):
        print('Quit! Already finish running for %s' % args.name)
        sys.exit()

    # Set seed
    if args.seed is not None:
        lib.utils.seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_train, X_valid, X_test, y_train, y_valid, y_test, problem, metric = \
        preprocess_data(args)

    args.in_features, args.problem = X_train.shape[1], problem
    args.num_classes = 1 # No matter classification / regression, use only 1 output

    # Model
    model, step_callback = getattr(lib.arch, args.arch + 'Block').load_model_by_hparams(
        args, ret_step_callback=True)
    model.to(device)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    from qhoptim.pyt import QHAdam
    optimizer_params = {'nus':(0.7, 1.0), 'betas':(0.95, 0.998)}

    loss_function = lambda x, y: F.binary_cross_entropy_with_logits(x, y.float()) \
        if args.problem == 'classification' else F.mse_loss
    trainer = lib.Trainer(
        model=model, loss_function=loss_function,
        experiment_name=args.name,
        warm_start=True, # To handle the interruption on v server
        Optimizer=QHAdam,
        optimizer_params=optimizer_params,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        verbose=False,
        n_last_checkpoints=5,
        step_callbacks=[step_callback] if step_callback is not None else [], # Temp annelaing
        l2_lambda=args.l2_lambda,
        fp16=args.fp16,
    )

    assert metric in ['negative_auc', 'classification_error', 'mse']
    eval_fn = getattr(trainer, 'evaluate_' + metric)

    # Before we start, we will need to select the batch size if unspecified
    if args.batch_size is None:
        assert device != 'cpu', 'Have to specify batch size when using CPU'
        args.batch_size = choose_batch_size(trainer, X_train, y_train, device,
                                            max_bs=args.max_bs, min_bs=args.min_bs)
        # Increase learning rate if bigger than 1024 batch size
        # args.lr = trainer.lr = np.ceil(args.batch_size / 1024) * args.lr
    else:
        with torch.no_grad():
            res = model(torch.as_tensor(X_train[:(2 * args.batch_size)], device=device))
            # trigger data-aware init

    # Then show hparams after deciding the batch size
    print("experiment:", args.name)
    print("Args:")
    print(args)

    # Then record hparams
    saved_args = pjoin('logs', args.name, 'hparams.json')
    json.dump(vars(args), open(saved_args, 'w'))

    # To make sure when rerunning the err history and time are accurate,
    # we save the whole history in training.json.
    recorder = lib.Recorder(path=pjoin('logs', args.name))

    st_time = time.time()
    for batch in lib.iterate_minibatches(X_train, y_train,
                                         batch_size=args.batch_size,
                                         shuffle=True, epochs=float('inf')):
        metrics = trainer.train_on_batch(*batch, device=device)

        recorder.loss_history.append(float(metrics['loss']))

        if trainer.step % args.report_frequency == 0:
            trainer.save_checkpoint()
            trainer.average_checkpoints(out_tag='avg')
            trainer.load_checkpoint(tag='avg')

            err = eval_fn(X_valid, y_valid,
                          device=device, batch_size=args.batch_size * 2)
            err = float(err) # To avoid annoying JSON unserializable bug

            if err < recorder.best_err:
                recorder.best_err = err
                recorder.best_step_err = trainer.step
                trainer.save_checkpoint(tag='best')
            recorder.err_history.append(err)
            recorder.step = trainer.step

            recorder.run_time += float(time.time() - st_time)
            st_time = time.time()

            recorder.save_record()

            trainer.load_checkpoint()  # last
            trainer.remove_old_temp_checkpoints()
            save_loss_fig(recorder.loss_history, recorder.err_history,
                          pjoin('loss_figs', f'{args.name}.jpg'))

            if trainer.step // args.report_frequency == 1:
                print("Step\tVal_Err\tTime(s)")
            print('%d\t%0.5f\t%d' % (trainer.step, err, recorder.run_time))

        if trainer.step > max(getattr(args, 'anneal_steps', -1), recorder.best_step_err) \
                + args.early_stopping_rounds:
            print('BREAK. There is no improvment for {} steps'.format(
                args.early_stopping_rounds))
            break

        if args.lr_decay_steps > 0 \
                and trainer.step > recorder.best_step_err + args.lr_decay_steps \
                and trainer.step > (recorder.lr_decay_step + args.lr_decay_steps):
            lr_before = trainer.lr
            trainer.decrease_lr(ratio=0.2, min_lr=1e-4)
            recorder.lr_decay_step = trainer.step
            print('LR: %.2e -> %.2e' % (lr_before, trainer.lr))

        if args.max_rounds > 0 and trainer.step > args.max_rounds:
            print('End. It reaches the maximum rounds %d' % args.max_rounds)
            break

        if recorder.run_time > args.max_time:
            print('End. It reaches the maximum run time %d (s)' % args.max_time)
            break

    print("Best step: ", recorder.best_step_err)
    print("Best Val Error: %0.5f" % recorder.best_err)

    max_step = trainer.step
    # Run test time
    trainer.load_checkpoint(tag='best')
    test_err = eval_fn(X_test, y_test,
                       device=device, batch_size=2*args.batch_size)
    print('Best step: ', trainer.step)
    print("Test Error rate: %0.5f" % test_err)

    # Save csv results
    results = dict()
    results['test_err'] = '%.5f' % test_err
    results['val_err'] = '%.5f' % recorder.best_err
    results['best_step_err'] = recorder.best_step_err
    results['max_step'] = max_step
    results['time(s)'] = '%d' % recorder.run_time
    results['fold'] = args.fold
    results['fp16'] = args.fp16
    results['batch_size'] = args.batch_size
    # Append the hyperparameters
    rs_hparams = getattr(lib.arch, args.arch + 'Block').get_model_specific_rs_hparams()
    for k in rs_hparams:
        results[k] = getattr(args, k)

    results = getattr(lib.arch, args.arch + 'Block').add_model_specific_results(results, args)
    results['name'] = args.name

    os.makedirs(f'results', exist_ok=True)
    lib.utils.output_csv(f'results/{args.dataset}_{args.arch}_new.csv', results)

    # Clean up
    if pexists(pjoin('is_running', args.name)):
        os.remove(pjoin('is_running', args.name))
    open(pjoin('logs', args.name, 'MY_IS_FINISHED'), 'a')


def preprocess_data(args):
    # Data
    data = lib.DATASETS[args.dataset.upper()](path='./data', fold=args.fold)
    preprocessor = lib.MyPreprocessor(
        cat_features=data.get('cat_features', None),
        y_normalize=(data['problem'] == 'regression'),
        random_state=1337, quantile_transform=True,
        quantile_noise=1e-4)

    # Transform dataset
    preprocessor.fit(data['X_train'], data['y_train'])
    X_train, y_train = preprocessor.transform(data['X_train'], data['y_train'])
    X_test, y_test = preprocessor.transform(data['X_test'], data['y_test'])
    if 'X_valid' in data and 'y_valid' in data:
        X_valid, y_valid = preprocessor.transform(data['X_valid'], data['y_valid'])
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=1377)

    # Save preprocessor
    with open(pjoin('logs', args.name, 'preprocessor.pkl'), 'wb') as op:
        pickle.dump(preprocessor, op)

    metric = data.get('metric', 'classification_error'
        if data['problem'] == 'classification' else 'mse')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, data['problem'], metric


def choose_batch_size(trainer, X_train, y_train, device,
                      max_bs=4096, min_free_mem=2200, min_bs=64):
    def clean_up_memory():
        for p in trainer.model.parameters():
            p.grad = None
        torch.cuda.empty_cache()

    # Starts with biggest batch size
    bs = max_bs

    shuffle_indices = np.random.permutation(X_train.shape[0])

    while True:
        try:
            if bs <= min_bs:
                raise RuntimeError('The batch size %d is smaller than mininum %d'
                                   % (bs, min_bs))
            print('Trying batch size %d ...' % bs)
            metrics = trainer.train_on_batch(
                X_train[shuffle_indices[:bs]], y_train[shuffle_indices[:bs]],
                device=device, update=False)

            # Check the memory
            # f = lib.get_gpu_stat('memory.free', device_id=0)
            # if f < min_free_mem: # Less than 1 GB?
            #     print('Memory free is only %d (MB). '
            #           'Should be at least %d (MB)' % (f, min_free_mem))
            #     bs = bs // 2
            #     continue
            break
        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise e

            print('| batch size %d failed.' % (bs))
            bs = bs // 2
            if bs <= min_bs:
                raise e
            continue
        finally:
            clean_up_memory()

    print('Choose batch size %d.' % (bs))
    return bs


def save_loss_fig(loss_history, err_history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # At last, save the loss figure
    plt.figure(figsize=[18, 6])
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Loss')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(err_history)
    plt.title('Error')
    plt.grid()
    plt.savefig(path, bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    args = get_args()

    if args.random_search == 0:
        # Create directory
        main(args)
        sys.exit()

    # Create a directory to record what is running
    os.makedirs('is_running', exist_ok=True)

    rs_hparams = getattr(lib.arch, args.arch + 'Block').get_model_specific_rs_hparams()

    orig_name = args.name
    for r in range(args.random_search):
        for _ in range(20): # Try 20 times if can't found, quit
            for k, v in rs_hparams.items():
                setattr(args, k, v['gen'](args))

            args.name = orig_name + '_' + get_rs_name(args, rs_hparams)

            lock = FileLock("rs.lock")
            with lock:
                if pexists(pjoin('is_running', args.name)) \
                        or pexists(pjoin('logs', args.name)):
                    continue
                Path(pjoin('is_running', args.name)).touch()

            if platform.node().startswith('vws'):
                main(args)
            else:
                rest_attrs = [
                    'arch', 'early_stopping_rounds', 'dataset',
                    'fold', 'fp16', 'max_rounds', 'max_time',
                    'report_frequency', 'batch_size', 'max_bs',
                    'min_bs',
                ]

                tmp = {k: getattr(args, k) for k in rest_attrs if getattr(args, k) is not None}
                tmp.update({k: getattr(args, k) for k in rs_hparams if getattr(args, k) is not None})
                os.system(
                    './my_sbatch --cpu {} --gpus {} --mem {} --name {} '
                    'python -u main.py {}'.format(
                        args.cpu,
                        args.gpus,
                        args.mem,
                        args.name,
                        " ".join([f'--{k} {v}' for k, v in tmp.items()]),
                    ))
            break
        else:
            print('Can not find any more parameters! Quit.')
            sys.exit()
