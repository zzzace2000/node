import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_mean(s):
    if isinstance(s, float):
        return s

    if isinstance(s, str):
        if ' +-' in s:
            s = s[:s.index(' +-')]
        return float(s)

    raise Exception('the input is wierd: %s' % str(s))

def rank(x, is_metric_higher_better=True):
    x = x.apply(extract_mean)
    return x.rank(method='average', ascending=(not is_metric_higher_better), na_option='bottom')

def normalized_score(x, is_metric_higher_better=True, min_value=None):
    x = x.apply(extract_mean)

    if min_value is None:
        min_value = x.min()

    score = (x - min_value) / (x.max() - min_value)
    if not is_metric_higher_better:
        score = 1. - score
    return score

def add_new_row(table, series, row_name):
    new_indexes = list(table.index) + [row_name]
    new_table = table.append(series, ignore_index=True)
    new_table.index = new_indexes
    return new_table

def highlight_min_max(x):
    x = x.apply(extract_mean)
    return ['background-color: #cd4f39' if v == np.nanmax(x) else ('background-color: lightgreen' if v == np.nanmin(x) else '') for v in x]


def plot_models(models_dict, feat_name=None, fig=None, ax=None):
    dfs_dict = {}
    x_values_lookup = None
    for name, model in models_dict.items():
        df = model.get_GAM_df(x_values_lookup)
        if x_values_lookup is None:
            x_values_lookup = df.set_index('feat_name')['x']

        dfs_dict[name] = df

    return plot_dfs(dfs_dict, feat_name, fig, ax)


def plot_dfs(dfs_dict, feat_name=None, feat_idx=None, fig=None, ax=None):
    if feat_name is None and feat_idx is None:
        model_name = next(iter(dfs_dict))
        feat_name = dfs_dict[model_name].loc[1].feat_name
        print('Feature is not specified. Just pick the most important feature %s for the model %s' 
            % (feat_name, model_name))

    if ax is None:
        fig, ax = plt.subplots()

    for name, df in dfs_dict.items():
        if feat_idx is not None:
            row = df[df.feat_idx == feat_idx].iloc[0]
        else:
            row = df[df.feat_name == feat_name].iloc[0]
        y_std = row.y_std if 'y_std' in row else 0.
        ax.errorbar(row.x, row.y, y_std, label=str(name))

    ax.set_title(feat_name)
    ax.legend()
    return fig, ax


def cal_statistics(table, is_metric_higher_better, add_ns_baseline=False):
    # Add two rows
    mean_score = table.apply(lambda x: x.apply(lambda s: float(s[:s.index(' +-')] if isinstance(s, str) and ' +-' in s else s)).mean(), axis=0)
    new_table = add_new_row(table, mean_score, 'average')
    
    average_rank = mean_score.rank(ascending=(not is_metric_higher_better))
    new_table = add_new_row(new_table, average_rank, 'average_rank')

    mean_rank = table.apply(rank, axis=1, is_metric_higher_better=is_metric_higher_better).mean()
    new_table = add_new_row(new_table, mean_rank.apply(lambda x: '%.2f' % x), 'avg_rank')
    
    avg_rank_rank = mean_rank.rank(ascending=True)
    new_table = add_new_row(new_table, avg_rank_rank, 'avg_rank_rank')

    mean_normalized_score = table.apply(normalized_score, axis=1, is_metric_higher_better=is_metric_higher_better).mean()
    new_table = add_new_row(new_table, mean_normalized_score.apply(lambda x: '%.3f' % x), 'avg_score')
    
    avg_score_rank = mean_normalized_score.rank(ascending=False)
    new_table = add_new_row(new_table, avg_score_rank, 'avg_score_rank')
    
    if add_ns_baseline:
        mean_normalized_score_b = table.apply(normalized_score, axis=1, min_value=0.5, is_metric_higher_better=is_metric_higher_better).mean()
        new_table = add_new_row(new_table, mean_normalized_score_b.apply(lambda x: '%.3f' % x), 'avg_score_b0.5')

        avg_score_rank_b = mean_normalized_score_b.rank(ascending=False)
        new_table = add_new_row(new_table, avg_score_rank_b, 'avg_score_b0.5_rank')

    return new_table


def vis_main_effects(all_dfs, num_cols=4, model_names=None, only_non_binary=False, call_backs=None, 
                     feature_names=None, figsize=None, removed_feature_names=None, 
                     vertical_margin=4, horizontal_margin=2, top_interactions=0,
                     only_interactions=False):
    if model_names is None:
        model_names = list(all_dfs.keys())
    else:
        all_dfs = {k: all_dfs[k] for k in model_names}

    first_df = all_dfs[next(iter(all_dfs))]
    first_df = first_df[first_df.feat_idx != -1] # Remove bias first
    if only_non_binary:
        first_df = first_df[first_df.x.apply(lambda x: x is not None and len(x) > 2)]

    if top_interactions >= 0:
        if top_interactions == 0:
            first_df = first_df[first_df.feat_idx.apply(lambda x: not isinstance(x, tuple))]
        else:
            df_main = first_df[first_df.feat_idx.apply(lambda x: not isinstance(x, tuple))]
            df_iter = first_df[first_df.feat_idx.apply(lambda x: isinstance(x, tuple))]
            df_iter = df_iter.sort_values('importance', ascending=False).iloc[:top_interactions]
            first_df = pd.concat([df_main, df_iter], axis=0)

    if only_interactions:
        first_df = first_df[first_df.feat_idx.apply(lambda x: isinstance(x, tuple))]

    if feature_names is not None:
        first_df = first_df[first_df.feat_name.apply(lambda x: x in feature_names)]

    if removed_feature_names is not None:
        first_df = first_df[first_df.feat_name.apply(lambda x: x not in removed_feature_names)]

    num_rows = int(np.ceil((len(first_df)) / num_cols))
    if figsize is None:
        figsize = (5 * num_cols + horizontal_margin * (num_cols - 1),
                   3 * num_rows + vertical_margin * (num_rows-1))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    the_df_lookups = {k: df.set_index('feat_idx') for k, df in all_dfs.items()}

    ax_idx = 0
    for r_idx, row in first_df.iterrows():
        the_ax = axes if not isinstance(axes, np.ndarray) else axes.flat[ax_idx]

        if isinstance(row.feat_idx, int): # main effect
            if isinstance(row.x[0], str): # categorical variable
                y_dfs = []
                yerr_dfs = []
                for model_name in model_names:
                    lookup = the_df_lookups[model_name]
                    y_df = pd.DataFrame(lookup.loc[row.feat_idx].y,
                                        index=lookup.loc[row.feat_idx].x,
                                        columns=[model_name])
                    y_dfs.append(y_df)

                    if 'y_std' not in lookup.loc[row.feat_idx]:
                        continue

                    yerr_df = pd.DataFrame(
                        lookup.loc[row.feat_idx].y_std,
                        index=lookup.loc[row.feat_idx].x,
                        columns=[model_name])
                    yerr_dfs.append(yerr_df)

                y_dfs = pd.concat(y_dfs, axis=1)
                if len(yerr_dfs) > 0:
                    yerr_dfs = pd.concat(yerr_dfs, axis=1)
                else:
                    yerr_dfs = None

                y_dfs.plot.bar(ax=the_ax, yerr=yerr_dfs)

                # Rotate back to 0
                for tick in the_ax.get_xticklabels():
                    tick.set_rotation(0)

                # if it's a boolean, set the rotation back
    #             if len(the_df_lookups[model_names[0]].loc[feat_name].x) == 2:
    #                 the_ax.set_xticklabels([0, 1])
    #                 for tick in the_ax.get_xticklabels():
    #                     tick.set_rotation(0)

                # sns.barplot(x='x', y='y', hue='model_name', data=all_plot_dfs, ax=the_ax)
                # for tick in the_ax.get_xticklabels():
                #     tick.set_rotation(45)
            else:
                for model_name in model_names:
                    if model_name not in all_dfs:
                        print('%s not in the all_dfs' % model_name)
                        continue

                    the_df_lookup = the_df_lookups[model_name]

                    y_std = 0 if 'y_std' not in the_df_lookup.loc[row.feat_idx] \
                        else the_df_lookup.loc[row.feat_idx].y_std

                    the_ax.errorbar(
                        the_df_lookup.loc[row.feat_idx].x,
                        the_df_lookup.loc[row.feat_idx].y,
                        y_std, label=model_name)
            the_ax.legend()
        else: # interaction effect
            # print('We only plot the first df in interaction term')
            all_x = [t[0] for t in row.x]
            all_y = [t[1] for t in row.x]
            # feat_names = [first_df[first_df.feat_idx == row.feat_idx[i]].feat_name.iloc[0]
            #               for i in range(2)]
            feat_names = row.feat_name.split('_')

            x_len, y_len = len(set(all_x)), len(set(all_y))
            if x_len > 4 and y_len > 4:
                # Plot the scatter plot
                # if x_len >= y_len:
                cax = sns.scatterplot(x=all_x, y=all_y, hue=row.y, palette='RdBu', ax=the_ax, s=50)
                the_ax.set_xlabel(feat_names[0])
                the_ax.set_ylabel(feat_names[1])
                # else:
                #     cax = sns.scatterplot(x=all_y, y=all_x, hue=row.y, palette='RdBu', ax=the_ax, s=50)
                #     the_ax.set_xlabel(feat_names[1])
                #     the_ax.set_ylabel(feat_names[0])

                vlim = np.max(np.abs(row.y))
                norm = plt.Normalize(-vlim, vlim)
                sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
                sm.set_array([])

                # Remove the legend and add a colorbar
                cax.get_legend().remove()
                # the_ax.get_legend().remove()
                cax.figure.colorbar(sm, ax=the_ax)

            elif x_len <= 4 and x_len < y_len:
                uniq_x, inv = np.unique(all_x, return_inverse=True)

                for i, x in enumerate(uniq_x):
                    y = np.array(all_y)[inv == i]
                    val = np.array(row.y)[inv == i]
                    val_std = 0.
                    if 'y_std' in row:
                        val_std = np.array(row.y_std)[inv == i]
                    the_ax.errorbar(y, val, val_std, label=f'{feat_names[0]}={x}')
                the_ax.set_xlabel(feat_names[1])
                the_ax.legend()
            else:
                uniq_y, inv = np.unique(all_y, return_inverse=True)

                for i, x in enumerate(uniq_y):
                    y = np.array(all_x)[inv == i]
                    val = np.array(row.y)[inv == i]
                    val_std = 0.
                    if 'y_std' in row:
                        val_std = np.array(row.y_std)[inv == i]
                    the_ax.errorbar(y, val, val_std, label=f'{feat_names[1]}={x}')
                the_ax.set_xlabel(feat_names[0])
                the_ax.legend()

        title = row.feat_name
        if 'importance' in row:
            title += ' (Imp=%.2e)' % row.importance
        the_ax.set_title(title)
        ax_idx += 1

        if call_backs is not None and row.feat_name in call_backs:
            call_backs[row.feat_name](the_ax)
    
    return fig, axes


# def vis_main_effects(all_dfs, num_cols=4, model_names=None, only_non_binary=False, call_backs=None, 
#                      feature_names=None, figsize=None, removed_feature_names=None):
#     first_df = all_dfs[next(iter(all_dfs))]
#     if only_non_binary:
#         first_df = first_df[first_df.x.apply(lambda x: x is not None and len(x) > 2)]
    
#     if feature_names is None:
#         feature_names = first_df.feat_name.unique()
#         feature_names = feature_names[feature_names != 'offset']
    
#     if removed_feature_names is not None:
#         feature_names = [f for f in feature_names if f not in removed_feature_names]
    
#     if model_names is None:
#         model_names = list(all_dfs.keys())

#     num_rows = int(np.ceil(len(feature_names) / num_cols))
#     if figsize is None:
#         figsize = (5 * num_cols + 2 * (num_cols - 1), 3 * num_rows + 2 * (num_rows-1))
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

#     the_df_lookups = {k: df.set_index('feat_name') for k, df in all_dfs.items()}

#     for f_idx, feat_name in enumerate(feature_names):
#         the_ax = axes if not isinstance(axes, np.ndarray) else axes.flat[f_idx]
#         if isinstance(the_df_lookups[model_names[0]].loc[feat_name].x[0], str):
#             # This line would include binary variable. It seems AIDS are too terrible :(
#             # or len(the_df_lookups[model_names[0]].loc[feat_name].x) == 2:

#             # categorical variable
#             y_dfs = []
#             yerr_dfs = []
#             for model_name in model_names:
#                 lookup = the_df_lookups[model_name]
#                 y_df = pd.DataFrame(lookup.loc[feat_name].y, index=lookup.loc[feat_name].x, columns=[model_name])
#                 y_dfs.append(y_df)

#                 if 'y_std' not in lookup.loc[feat_name]:
#                     continue

#                 yerr_df = pd.DataFrame(lookup.loc[feat_name].y_std, index=lookup.loc[feat_name].x, columns=[model_name])
#                 yerr_dfs.append(yerr_df)

#             y_dfs = pd.concat(y_dfs, axis=1)
#             if len(yerr_dfs) > 0:
#                 yerr_dfs = pd.concat(yerr_dfs, axis=1)
#             else:
#                 yerr_dfs = None
            
#             y_dfs.plot.bar(ax=the_ax, yerr=yerr_dfs)

#             # if it's a boolean, set the rotation back
#             if len(the_df_lookups[model_names[0]].loc[feat_name].x) == 2:
#                 the_ax.set_xticklabels(the_df_lookups[model_names[0]].loc[feat_name].x)
#                 for tick in the_ax.get_xticklabels():
#                     tick.set_rotation(0)

#             # sns.barplot(x='x', y='y', hue='model_name', data=all_plot_dfs, ax=the_ax)
#             # for tick in the_ax.get_xticklabels():
#             #     tick.set_rotation(45)
#         else:
#             for model_name in model_names:
#                 if model_name not in all_dfs:
#                     print('%s not in the all_dfs' % model_name)
#                     continue
                
#                 the_df_lookup = the_df_lookups[model_name]
                
#                 y_std = 0 if 'y_std' not in the_df_lookup.loc[feat_name] else the_df_lookup.loc[feat_name].y_std

#                 the_ax.errorbar(the_df_lookup.loc[feat_name].x, the_df_lookup.loc[feat_name].y, y_std, label=model_name)

#         the_ax.set_title(feat_name)
#         the_ax.legend()
    
#         if call_backs is not None and feat_name in call_backs:
#             call_backs[feat_name](the_ax)
    
#     return fig, axes
    