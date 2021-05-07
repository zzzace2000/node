import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .nn_utils import sparsemax, sparsemoid, ModuleWithInit, entmax15, entmoid15
from .utils import check_numpy
from warnings import warn


MIN_LOGITS = -10


class ODST(ModuleWithInit):
    def __init__(self, in_features, num_trees, depth=6, tree_dim=1, flatten_output=True,
                 choice_function=entmax15, bin_function=entmoid15,
                 initialize_response_=nn.init.normal_,
                 initialize_selection_logits_=nn.init.uniform_,
                 threshold_init_beta=1.0, threshold_init_cutoff=1.0,
                 colsample_bytree=1., save_memory=True,
                 ):
        """
        Oblivious Differentiable Sparsemax Trees. http://tinyurl.com/odst-readmore
        One can drop (sic!) this module anywhere instead of nn.Linear
        :param in_features: number of features in the input tensor
        :param num_trees: number of trees in this layer
        :param tree_dim: number of response channels in the response of individual tree
        :param depth: number of splits in every tree
        :param flatten_output: if False, returns [..., num_trees, tree_dim],
            by default returns [..., num_trees * tree_dim]
        :param choice_function: f(tensor, dim) -> R_simplex computes feature weights s.t. f(tensor, dim).sum(dim) == 1
        :param bin_function: f(tensor) -> R[0, 1], computes tree leaf weights

        :param initialize_response_: in-place initializer for tree output tensor
        :param initialize_selection_logits_: in-place initializer for logits that select features for the tree
        both thresholds and scales are initialized with data-aware init (or .load_state_dict)
        :param threshold_init_beta: initializes threshold to a q-th quantile of data points
            where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
            If this param is set to 1, initial thresholds will have the same distribution as data points
            If greater than 1 (e.g. 10), thresholds will be closer to median data value
            If less than 1 (e.g. 0.1), thresholds will approach min/max data values.

        :param threshold_init_cutoff: threshold log-temperatures initializer, \in (0, inf)
            By default(1.0), log-remperatures are initialized in such a way that all bin selectors
            end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
            Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
            Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
            For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
            Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
            All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)

        :param colsample_bytree: the same argument as in xgboost package.
            If less than 1, for each tree, it will only choose a fraction of features to train. For instance,
            if colsample_bytree = 0.9, each tree will only selects among 90% of the features.
        """
        super().__init__()
        self.in_features, self.depth, self.num_trees, self.tree_dim, self.flatten_output = \
            in_features, depth, num_trees, tree_dim, flatten_output
        self.choice_function, self.bin_function = choice_function, bin_function
        self.threshold_init_beta, self.threshold_init_cutoff = threshold_init_beta, threshold_init_cutoff
        self.colsample_bytree = colsample_bytree

        self.response = nn.Parameter(torch.zeros([num_trees, tree_dim, 2 ** depth]), requires_grad=True)
        initialize_response_(self.response)

        self.num_sample_feats = in_features
        if self.colsample_bytree < 1.:
            self.num_sample_feats = int(np.ceil(in_features * self.colsample_bytree))

        # Do the subsampling
        if self.num_sample_feats < in_features:
            self.colsample = nn.Parameter(
                torch.zeros([in_features, num_trees, 1]), requires_grad=False
            )
            for nt in range(num_trees):
                rand_idx = torch.randperm(in_features)[:self.num_sample_feats]
                self.colsample[rand_idx, nt, 0] = 1.

        if self.num_sample_feats > 1 or (not save_memory):
            # To save the memory!
            self.feature_selection_logits = nn.Parameter(
                torch.zeros([in_features, num_trees, depth]), requires_grad=True
            )
            initialize_selection_logits_(self.feature_selection_logits)

        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True
        )  # nan values will be initialized on first batch (data-aware init)

        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True
        )

        # binary codes for mapping between 1-hot vectors and bin indices
        with torch.no_grad():
            indices = torch.arange(2 ** self.depth)
            offsets = 2 ** torch.arange(self.depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)
            # ^-- [depth, 2 ** depth, 2]

    def forward(self, input):
        assert len(input.shape) >= 2
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)
        # new input shape: [batch_size, in_features]

        feature_values = self.get_feature_selection_values(input)
        # ^--[batch_size, num_trees, depth]

        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(-self.log_temperatures)

        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        # ^--[batch_size, num_trees, depth, 2]

        bins = self.bin_function(threshold_logits)
        # ^--[batch_size, num_trees, depth, 2], approximately binary

        bin_matches = torch.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)
        # ^--[batch_size, num_trees, depth, 2 ** depth]

        response_weights = torch.prod(bin_matches, dim=-2)
        # ^-- [batch_size, num_trees, 2 ** depth]

        response = torch.einsum('bnd,ncd->bnc', response_weights, self.response)
        # ^-- [batch_size, num_trees, tree_dim]

        return response.flatten(1, 2) if self.flatten_output else response

    def initialize(self, input, eps=1e-6):
        # data-aware initializer
        assert len(input.shape) == 2
        if input.shape[0] < 1000:
            warn("Data-aware initialization is performed on less than 1000 data points. This may cause instability."
                 "To avoid potential problems, run this model on a data batch with at least 1000 data samples."
                 "You can do so manually before training. Use with torch.no_grad() for memory efficiency.")
        with torch.no_grad():
            feature_values = self.get_feature_selection_values(input)
            # ^--[batch_size, num_trees, depth]

            # initialize thresholds: sample random percentiles of data
            percentiles_q = 100 * np.random.beta(self.threshold_init_beta, self.threshold_init_beta,
                                                 size=[self.num_trees, self.depth])
            self.feature_thresholds.data[...] = torch.as_tensor(
                list(map(np.percentile, check_numpy(feature_values.flatten(1, 2).t()), percentiles_q.flatten())),
                dtype=feature_values.dtype, device=feature_values.device
            ).view(self.num_trees, self.depth)

            # init temperatures: make sure enough data points are in the linear region of sparse-sigmoid
            temperatures = np.percentile(check_numpy(abs(feature_values - self.feature_thresholds)),
                                         q=100 * min(1.0, self.threshold_init_cutoff), axis=0)

            # if threshold_init_cutoff > 1, scale everything down by it
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures) + eps)

    def get_feature_selection_values(self, input):
        ''' Select which features to split '''
        feature_selectors = self.get_feature_selectors()
        # ^--[in_features, num_trees, depth]

        feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)
        return feature_values

    def get_feature_selectors(self):
        if self.colsample_bytree < 1. and self.num_sample_feats == 1:
            return self.colsample.data

        fsl = self.feature_selection_logits
        if self.colsample_bytree < 1.:
            fsl = self.colsample * fsl \
                  + (1. - self.colsample) * MIN_LOGITS
        feature_selectors = self.choice_function(fsl, dim=0)
        return feature_selectors

    def __repr__(self):
        return "{}(in_features={}, num_trees={}, depth={}, tree_dim={}, flatten_output={})".format(
            self.__class__.__name__, self.in_features,
            self.num_trees, self.depth, self.tree_dim, self.flatten_output
        )


class GAM_ODST(ODST):
    def __init__(self, *args, selectors_detach=False, fs_normalize=True,
                 initialize_selection_logits_=nn.init.uniform_,
                 save_memory=True, ga2m=0, **kwargs):
        super().__init__(
            *args,
            initialize_selection_logits_=initialize_selection_logits_,
            save_memory=save_memory,
            **kwargs)
        self.selectors_detach = selectors_detach
        self.fs_normalize = fs_normalize
        self.ga2m = ga2m
        if ga2m:
            assert self.depth >= 2, f'depth should be > 2. depth={self.depth}'
            assert self.num_sample_feats > 1, \
                f'should sample > 1 feats. But get {self.num_sample_feats}.'

        # if more than 1 feature is subsampled, the initialize selection logits
        if self.num_sample_feats > 1 or (not save_memory):
            del self.feature_selection_logits

            the_depth = 1 if not self.ga2m else 2
            self.feature_selection_logits = nn.Parameter(
                torch.zeros([self.in_features, self.num_trees, the_depth]), requires_grad=True
            )
            initialize_selection_logits_(self.feature_selection_logits)

    def forward(self, input, return_feature_selectors=True, prev_feature_selectors=None):
        self.prev_feature_selectors = prev_feature_selectors

        response = super().forward(input)

        fs, self.feature_selectors = self.feature_selectors, None
        if return_feature_selectors:
            return response, fs

        return response

    def initialize(self, input, return_feature_selectors=True,
                   prev_feature_selectors=None, eps=1e-6):
        self.prev_feature_selectors = prev_feature_selectors
        response = super().initialize(input, eps=eps)
        self.feature_selectors = None

    def get_feature_selection_values(self, input):
        feature_selectors = self.get_feature_selectors()
        # ^--[in_features, num_trees, depth=1]

        # A hack to pass this value outside of this function
        self.feature_selectors = feature_selectors
        if self.selectors_detach: # To save memory
            self.feature_selectors = self.feature_selectors.detach()

        if self.ga2m:
            self.feature_selectors = self.feature_selectors.sum(dim=-1, keepdim=True)

        # It needs to multiply by the tree_dim
        if self.tree_dim > 1:
            self.feature_selectors = self.feature_selectors.repeat(
                1, 1, self.tree_dim,
            )
        self.feature_selectors = self.feature_selectors.view(self.in_features, -1)
        # ^--[in_features, num_trees * tree_dim]

        if input.shape[1] > self.in_features:  # The rest are previous layers
            # Check incoming data
            pfs, self.prev_feature_selectors = self.prev_feature_selectors, None
            assert pfs.shape == (self.in_features, input.shape[1] - self.in_features), \
                'Previous selectors does not have the same shape as the input: %s != %s' \
                % (pfs.shape, (self.in_features, input.shape[1] - self.in_features))
            fw = self.cal_prev_feat_weights(feature_selectors, pfs)

            feature_selectors = torch.cat([feature_selectors, fw], dim=0)
            # ^--[input_features, num_trees, depth=1]

        # post_process it
        feature_selectors = self.post_process(feature_selectors)
        # if self.fs_normalize:
        #     feature_selectors /= feature_selectors.sum(dim=0, keepdims=True)

        fv = torch.einsum('bi,ind->bnd', input, feature_selectors)
        # ^--[batch_size, num_trees, depth=1,2]
        if not self.ga2m:
            fv = fv.expand(-1, -1, self.depth)
        else:
            if self.depth > 2:
                fv = fv.repeat(1, 1, int(np.ceil(self.depth / 2)))[..., :self.depth]
        return fv
    #
    # def get_feature_selectors(self):
    #     fs = super().get_feature_selectors()
    #     # ^--[in_features, num_trees, 1]
    #
    #     fs = fs.expand(*fs.shape[:2], self.depth)
    #     # ^--[in_features, num_trees, depth]
    #     return fs

    def cal_prev_feat_weights(self, feature_selectors, pfs):
        # Do a row-wise inner product between prev selectors and cur ones
        if not self.ga2m:
            fw = torch.einsum('ip,icd->pcd', pfs, feature_selectors)
        else:
            p1 = torch.einsum('ic,ip->pc', feature_selectors[:, :, 0], pfs)
            p2 = torch.einsum('ic,ip->pc', feature_selectors[:, :, 1], pfs)

            fw = (p1 * p2).unsqueeze_(-1)
            fw = fw.repeat(1, 1, 2)
        return fw

    def post_process(self, feature_selectors):
        result = feature_selectors
        if self.fs_normalize:
            result = (feature_selectors / feature_selectors.sum(dim=0, keepdims=True))
        return result

    def get_num_trees_assigned_to_each_feature(self):
        with torch.no_grad():
            fs = self.get_feature_selectors()
            # ^-- [in_features, num_trees, 1]
            return (fs > 0).sum(dim=[1, 2])

    def get_interactions(self):
        if not self.ga2m:
            return None

        with torch.no_grad():
            fs = self.get_feature_selectors()
            # ^-- [in_features, num_trees, 2]
            tmp = fs.sum(dim=-1).T
            # ^-- [num_trees, in_features]
            r_idx, c_idx = torch.nonzero(tmp, as_tuple=True)

            interactions = []
            for r in range(self.num_trees):
                n_interaction = (r_idx == r).sum()
                if n_interaction > 1:
                    interactions.append(c_idx[r_idx == r][:2])
                if n_interaction > 2:
                    print(f'WARNING: it is not a GA2M with a {n_interaction}-way term. '
                          f'Only choose the first 2.')
            if len(interactions) == 0:
                return None

            interactions = torch.stack(interactions, dim=0)
            return torch.unique(interactions, dim=0)


class GAMAtt3ODST(GAM_ODST):
    def __init__(self, *args,
                 prev_in_features=0,
                 dim_att=-1,
                 initialize_selection_logits_=nn.init.uniform_,
                 **kwargs):
        super().__init__(*args,
                         initialize_selection_logits_=initialize_selection_logits_,
                         **kwargs)
        self.prev_in_features = prev_in_features
        self.dim_att = dim_att

        if prev_in_features > 0:
            self.att_key = nn.Parameter(
                torch.zeros([self.in_features + prev_in_features, dim_att]),
                requires_grad=True
            )
            self.att_query = nn.Parameter(
                torch.zeros([dim_att, self.num_trees]), requires_grad=True
            )
            initialize_selection_logits_(self.att_key)
            initialize_selection_logits_(self.att_query)

    def post_process(self, feature_selectors):
        fs = feature_selectors
        if self.prev_in_features > 0:
            pfa = torch.einsum('pa,at->pt', self.att_key, self.att_query)
            fs = entmax15(fs.add_(1e-20).log_().add_(pfa.unsqueeze_(-1)), dim=0)
        return fs

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict)


class GAMAttODST(GAM_ODST):
    def __init__(self, *args,
                 prev_in_features=0,
                 dim_att=-1,
                 initialize_selection_logits_=nn.init.uniform_,
                 **kwargs):
        kwargs['fs_normalize'] = False # No need normalization
        super().__init__(*args,
                         initialize_selection_logits_=initialize_selection_logits_,
                         **kwargs)

        self.prev_in_features = prev_in_features
        self.dim_att = dim_att
        if prev_in_features > 0:
            # Save parameter
            self.att_key = nn.Parameter(
                torch.zeros([prev_in_features, dim_att]), requires_grad=True
            )
            self.att_query = nn.Parameter(
                torch.zeros([dim_att, self.num_trees]), requires_grad=True
            )
            initialize_selection_logits_(self.att_key)
            initialize_selection_logits_(self.att_query)

    def cal_prev_feat_weights(self, feature_selectors, pfs):
        assert self.prev_in_features > 0

        fw = super().cal_prev_feat_weights(feature_selectors, pfs)
        # ^--[prev_in_feats, num_trees, 1]

        pfa = torch.einsum('pa,at->pt', self.att_key, self.att_query)
        fw = entmax15(fw.add_(1e-20).log_().add_(pfa.unsqueeze_(-1)), dim=0)
        return fw
