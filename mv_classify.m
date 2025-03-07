function [perf, result, testlabel] = mv_classify(cfg, X, clabel, varargin)
% Classification of multi-dimensional data.
%
% mv_classify allows for the classification of data of arbitrary number and
% order of dimensions. It can perform searchlight analysis and
% generalization. 
%
% Note: For data that has the dimensions [samples x features x time points], 
% you can alternatively use the more specialized functions 
% mv_classify_across_time and mv_classify_timextime. They have presets like
% the dimension names and are slightly faster. They provide the same
% numerical results as mv_classify.
%
% Usage:
% [perf, res] = mv_classify(cfg, X, clabel, <X2, clabel2>)
%
%Parameters:
% X              - [... x ... x ... x ] data matrix or kernel matrix of
%                  arbitrary dimensions
% clabel         - [samples x 1] vector of class labels
% X2, clabel2    - (optional) if a second dataset is provided, transfer
%                  classification (aka cross decoding) is performed. 
%                  X/clabel acts as train data and X2/clabel2 acts as test 
%                  data. The datasets must have the same size, they can
%                  only differ in the number of samples and in the
%                  generalization dimension.
%
% cfg          - struct with optional parameters:
% .classifier     - name of classifier, needs to have according train_ and test_
%                   functions (default 'lda')
% .hyperparameter - struct with parameters passed on to the classifier train
%                   function (default [])
% .metric         - classifier performance metric, default 'accuracy'. See
%                   mv_classifier_performance. If set to [] or 'none', the 
%                   raw classifier output (labels, dvals or probabilities 
%                   depending on cfg.output_type) for each sample is returned. 
%                   Use cell array to specify multiple metrics (eg
%                    {'accuracy' 'auc'}
% .feedback       - print feedback on the console (default 1)
% .save           - use to save labels or model parameters for each train iteration.
%                   The cell array can contain  'trainlabel', 'model_param' 
%                   (ie classifier parameters) (default {}). 
%                   The results struct then contains the eponymous fields.
%
% For mv_classify to make sense of the data, the user must specify the
% meaning of each dimension. sample_dimension and feature_dimension
% specify which dimension(s) code for samples and features, respectively.
% All other dimensions will be treated as 'search' dimensions and a
% separate classification will be performed for each element of these
% dimensions. Example: let the data matrix be [samples x time x features x
% frequencies]. Let sample_dimension=1 and feature_dimension=3. The output
% of mv_classify will then be a [time x frequencies] (corresponding to
% dimensions 2 and 4) matrix of classification results.
% To use generalization (e.g. time x time, or frequency x frequency), set
% the generalization_dimension parameter. 
% 
% .sample_dimension  - the data dimension(s) that code the samples (default 1). 
%                      It has either one element or two elements when the
%                      data provided is a kernel matrix. 
%                      It cannot have more than 2 elements.
% .feature_dimension - the data dimension(s) that code the features (default
%                      2). There can be more than 1 feature dimension, but
%                      then a classifier must be used that can deal with
%                      multi-dimensional inputs. If a kernel matrix is
%                      provided, there cannot be a feature dimension.
% .generalization_dimension - any of the other (non-sample, non-feature) 
%                             dimensions can be used for generalization. 
%                             In generalization, a model is trained for each
%                             generalization element and then tested at
%                             each other element (default []). Note: if a 
%                             generalization dimension is given, the input
%                             may not consist of *precomputed kernels*.
%                             This is because the kernel matrix needs to be
%                             evaluated not only between samples within a
%                             given time point but also for all
%                             combinations of samples across different time
%                             points.
% .flatten_features  - if there is multiple feature dimensions, flattens
%                      the features into a single feature vector so that it
%                      can be used with the standard classifiers (default
%                      1). Has no effect if there is only one feature
%                      dimension.
% .dimension_names   - cell array with names for the dimensions. These names
%                      are used when printing the classification
%                      info.
%
% SEARCHLIGHT parameters:
% Every non-sample, non-feature dimension is designated as a 'searchlight
% dimension'. For instance, if the data is [samples x features x time], the
% time dimension usually serves as a searchlight dimension. This means that
% a separate classification is performed for each element of the
% searchlight dimension. Normally, every element is considered on its own,
% but the neighbours field can be used to consider multiple elements at
% once. 
% .neighbours  - [... x ...] matrix specifying which elements of the search
%                are neighbours of each other. If there is multiple search
%                dimensions, a cell array of such matrices should be
%                provided. (default: identity matrix). Note: this
%                corresponds to the GRAPH option in mv_searchlight.
%                There is no separate parameter for neighbourhood size, the
%                size of the neighbourhood is specified by the matrix.
% append       - if true, searchlight dimensions are appended to the train 
%                data instead of
%                being looped over. This can be useful for performance, 
%                since the model receives the whole data which obliterates 
%                the loop. For instance, if the data is 
%                [samples x features x time] and cfg.append = 1, 
%                the model receives a whole [train_samples x features x time] 
%                array at once, instead of the usual [train_samples x features] 
%                array. This also obliterates the loop across time. Note: 
%                at the moment only naive_bayes supports this feature.
%                (default false)
%
% CROSS-VALIDATION parameters:
% .cv           - perform cross-validation, can be set to 'kfold',
%                 'leaveout', 'holdout', 'predefined' or 'none' (default 'kfold')
% .k            - number of folds in k-fold cross-validation (default 5)
% .p            - if cv is 'holdout', p is the fraction of test samples
%                 (default 0.1)
% .stratify     - if 1, the class proportions are approximately preserved
%                 in each fold (default 1)
% .repeat       - number of times the cross-validation is repeated with new
%                 randomly assigned folds (default 1)
% .fold         - if cv='predefined', fold is a vector of length
%                 #samples that specifies the fold each sample belongs to
%
% PREPROCESSING parameters:  
% .preprocess         - cell array containing the preprocessing pipeline. The
%                       pipeline is applied in chronological order
% .preprocess_param   - cell array of preprocessing parameter structs for each
%                       function. Length of preprocess_param must match length
%                       of preprocess
%
% Returns:
% perf          - matrix of classification performances corresponding to 
%                 the selected metric. If multiple metrics are requested, 
%                 perf is a cell array
% result        - struct with fields describing the classification result.
%                 Can be used as input to mv_statistics and mv_plot_result
% testlabel     - cell array of test labels. Can be useful if metric='none'

X = double(X);

mv_set_default(cfg,'classifier','lda');
mv_set_default(cfg,'hyperparameter',[]);
mv_set_default(cfg,'metric','accuracy');
mv_set_default(cfg,'feedback',1);
mv_set_default(cfg,'save',{});

mv_set_default(cfg,'sample_dimension', 1);
mv_set_default(cfg,'generalization_dimension',[]);
if isempty(cfg.generalization_dimension) || cfg.generalization_dimension ~= 2
    mv_set_default(cfg,'feature_dimension', 2);
else
    mv_set_default(cfg,'feature_dimension', []);
end
mv_set_default(cfg,'append', false);
mv_set_default(cfg,'flatten_features',1);
mv_set_default(cfg,'dimension_names',strcat('dim', arrayfun(@(x) {num2str(x)}, 1:ndims(X))));

mv_set_default(cfg,'neighbours',{});
if isempty(cfg.neighbours), cfg.neighbours = {}; end  % replace [] by {}
if ~iscell(cfg.neighbours), cfg.neighbours = {cfg.neighbours}; end
cfg.neighbours = cfg.neighbours(:);  % make sure it's a column vector

mv_set_default(cfg,'preprocess',{});
mv_set_default(cfg,'preprocess_param',{});

has_second_dataset = (nargin==5);
if has_second_dataset
    X2 = double(varargin{1});
    [cfg, clabel, n_classes, n_metrics, clabel2] = mv_check_inputs(cfg, permute(X,[cfg.sample_dimension, setdiff(1:ndims(X), cfg.sample_dimension)]), clabel, permute(X2,[cfg.sample_dimension, setdiff(1:ndims(X), cfg.sample_dimension)]), varargin{2});
else
    [cfg, clabel, n_classes, n_metrics] = mv_check_inputs(cfg, permute(X,[cfg.sample_dimension, setdiff(1:ndims(X), cfg.sample_dimension)]), clabel);
end

% sort dimension vectors
sample_dim = sort(cfg.sample_dimension);
feature_dim = sort(cfg.feature_dimension);
gen_dim = cfg.generalization_dimension;

% define non-sample/feature dimension(s) that will be used for search/looping
search_dim = setdiff(1:ndims(X), [sample_dim, feature_dim]);

% Number of samples in the classes
n = arrayfun( @(c) sum(clabel==c) , 1:n_classes);

% indicates whether the data represents kernel matrices
mv_set_default(cfg,'is_kernel_matrix', isfield(cfg.hyperparameter,'kernel') && strcmp(cfg.hyperparameter.kernel,'precomputed'));

% generalization does not work together with precomputed kernel matrices
if ~isempty(gen_dim)
    assert(~cfg.is_kernel_matrix, 'generalization does not work together with precomputed kernel matrices')
    assert(any(ismember(gen_dim, search_dim)),'generalization dimension must be one of the search dimensions (different from sample and feature dimensions)')
end

assert(isempty(gen_dim) || ~cfg.append, 'generalization does not work together with appended dimensions')

if has_second_dataset
    sz1 = size(X);
    sz2 = size(X2);
    sz1([sample_dim gen_dim])=[]; sz2([sample_dim gen_dim]) = [];
    assert(all(sz1==sz2), sprintf('both datasets may only differ in their sample and generalization dimensions, but size(X) = [%s] and size(X2) = [%s]', num2str(size(X)), num2str(size(X2))))
end

if cfg.feedback, mv_print_classification_info(cfg,X,clabel, varargin{:}); end

%% check dimension parameters
% check sample dimensions
assert(numel(sample_dim)<=2, sprintf('There can be at most 2 sample dimensions but %d have been specified', numel(sample_dim)))
assert((numel(sample_dim)~=2) || cfg.is_kernel_matrix, 'there is 2 sample dimensions given but the kernel is not specified to be precomputed (set cfg.hyperparameter.kernel=''precomputed'')')
assert((numel(sample_dim)~=2) || (numel(feature_dim)==0), 'if there is 2 samples dimensions you must set cfg.feature_dimensions=[]')
assert(numel(gen_dim) <= 1, 'There can be at most one generalization dimension')

% check whether dimensions are different and add up to ndims(X)
sam_feat_gen_dims = sort([sample_dim, feature_dim, gen_dim]);
if numel(unique(sam_feat_gen_dims)) < numel(sam_feat_gen_dims)
    error('sample_dimension, feature_dimension, and generalization_dimension must be different from each other')
end

%% check neighbours parameters
has_neighbours = ~isempty(cfg.neighbours);
assert(~(has_neighbours && (numel(cfg.neighbours) ~= numel(search_dim))), 'If any neighbourhood matrix is given, you must specify a matrix for every search dimension')
assert(~(has_neighbours && numel(gen_dim)>0), 'Searchlight and generalization are currently not supported simultaneously')

%% order the dimensions by samples -> search dimensions -> features

if ~isempty(gen_dim) && (search_dim(end) ~= gen_dim)
    % the generalization dimension should be the last of the search dimensions,
    % if it is not then permute the dimensions accordingly
    ix = find(ismember(search_dim, gen_dim));
    % push gen dim to the end
    search_dim = [search_dim(1:ix-1), search_dim(ix+1:end), search_dim(ix)];
end

% permute X and dimension names
new_dim_order = [sample_dim, search_dim, feature_dim];
X = permute(X, new_dim_order);
if has_second_dataset, X2 = permute(X2, new_dim_order); end
cfg.dimension_names = cfg.dimension_names(new_dim_order);

% adapt the dimensions to reflect the permuted X
sample_dim = 1:numel(sample_dim);
search_dim = (1:numel(search_dim))  + numel(sample_dim);
feature_dim = (1:numel(feature_dim))+ numel(sample_dim) + numel(search_dim);
if ~isempty(gen_dim), gen_dim = search_dim(end); end

%% flatten features to one dimension if requested
if numel(feature_dim) > 1 && cfg.flatten_features
    sz_search = size(X);
    all_feat = prod(sz_search(feature_dim));
    X = reshape(X, [sz_search(sample_dim), sz_search(search_dim), all_feat]);
    if has_second_dataset
        sz_search2 = size(X2);
        X2 = reshape(X2, [sz_search2(sample_dim), sz_search2(search_dim), all_feat]); 
    end
    % also flatten dimension names
    cfg.dimension_names{feature_dim(1)} = strjoin(cfg.dimension_names(feature_dim),'/');
    cfg.dimension_names(feature_dim(2:end)) = [];
    feature_dim = feature_dim(1);
end

% rearrange dimensions in preprocess fields according to the new dimension order
cfg.preprocess_param = mv_rearrange_preprocess_dimensions(cfg.preprocess_param, new_dim_order, ndims(X));

%% Get train and test functions
train_fun = eval(['@train_' cfg.classifier]);
test_fun = eval(['@test_' cfg.classifier]);

% Define search dimension
if has_neighbours
    % size of the search dimension corresponds to the rows of the
    % neighbourhood matrices
    sz_search = cell2mat(cellfun(@(neigh) size(neigh,1), cfg.neighbours, 'Un', 0))';
else
    % size of the search dimensions is equal to size of the corresponding X
    % dimensions
    sz_search = size(X);
    sz_search = sz_search(search_dim);
    if isempty(sz_search), sz_search = 1; end    
end

% sample_skip and feature_skip helps us access the search dimensions by 
% skipping over sample(incl appending) and feature dimensions
% sample_skip = repmat({':'},[1, numel([sample_dim, feature_dim])] );
sample_skip = repmat({':'},[1, numel(sample_dim)] );
feature_skip = repmat({':'},[1, numel(feature_dim)] );

%% Create all combinations of elements in the search dimensions
if isempty(search_dim)
    % no search dimensions, so we just perform cross-validation once
    dim_loop = {':'};
elseif cfg.append
    % search dimensions are appended, so we just perform cross-validation once
    dim_loop = {':'};
    cfg.hyperparameter.neighbours       = cfg.neighbours;
    cfg.hyperparameter.is_multivariate  =  ~isempty(cfg.feature_dimension);    
else
    len_loop = prod(sz_search);
    dim_loop = zeros(numel(sz_search), len_loop);
    for rr = 1:numel(sz_search)  % row
        seq = mv_repelem(1:sz_search(rr), prod(sz_search(1:rr-1)));
        dim_loop(rr, :) = repmat(seq, [1, len_loop/numel(seq)]);
    end
    
    % to use dim_loop for indexing, we need to convert it to a cell array
    dim_loop = num2cell(dim_loop);
end

nfeat = [size(X) ones(1, numel(cfg.dimension_names) - ndims(X))];
nfeat = nfeat(feature_dim);
if isempty(nfeat), nfeat = 1; end

%% prepare save
if ~iscell(cfg.save), cfg.save = {cfg.save}; end
save_model = any(strcmp(cfg.save, 'model_param'));
save_trainlabel = any(strcmp(cfg.save, 'trainlabel'));

%% Perform classification
if ~strcmp(cfg.cv,'none') && ~has_second_dataset
    % -------------------------------------------------------
    % Perform cross-validation

    % Initialize classifier outputs
    if cfg.append
        cf_output = cell([cfg.repeat, cfg.k]);
    else
        cf_output = cell([cfg.repeat, cfg.k, sz_search]);
    end
    testlabel = cell([cfg.repeat, cfg.k]);
    if save_trainlabel, all_trainlabel = cell([cfg.repeat, cfg.k]); end
    if save_model, all_model = cell(size(cf_output)); end

    for rr=1:cfg.repeat                 % ---- CV repetitions ----
        if cfg.feedback, fprintf('Repetition #%d. Fold ',rr), end
        
        % Define cross-validation
        CV = mv_get_crossvalidation_folds(cfg.cv, clabel, cfg.k, cfg.stratify, cfg.p, cfg.fold, cfg.preprocess, cfg.preprocess_param);
        
        for kk=1:CV.NumTestSets                      % ---- CV folds ----
            if cfg.feedback
                if kk<=20, fprintf('%d ',kk), % print first 20 folds
                elseif kk==21, fprintf('... ') % then ... and stop to not spam the console too much
                elseif kk>CV.NumTestSets-5, fprintf('%d ',kk) % then the last 5 ones
                end
            end

            % Get train and test data
            [cfg, Xtrain, trainlabel, Xtest, testlabel{rr,kk}] = mv_select_train_and_test_data(cfg, X, clabel, CV.training(kk), CV.test(kk), cfg.is_kernel_matrix);

            if ~isempty(cfg.preprocess)
                % TODO: cfg_preproc = mv_select_preprocessing_data(pparam, ...) => select
                % from every field in pparam starting with
                % have pparam.data = {'signal' 'noise'}
                % cfg.preprocess_param.X_foo becomes X_foo_train and X_foo_test 
                % Preprocess train data
                [tmp_cfg, Xtrain, trainlabel] = mv_preprocess(cfg, Xtrain, trainlabel);
                
                % Preprocess test data
                [~, Xtest, testlabel{rr,kk}] = mv_preprocess(tmp_cfg, Xtest, testlabel{rr,kk});
            end
            
            if ~isempty(gen_dim)
                % ---- Generalization ---- (eg time x time)
                % Instead of looping through the generalization dimension,
                % which would require an additional loop, we reshape the test
                % data and apply the classifier to all elements of the
                % generalization dimension at once
                
                % gen_dim is the last search dimension. For reshaping we
                % need to move it to the first search position and
                % shift the other dimensions up one position
                Xtest = permute(Xtest, [sample_dim, search_dim(end), search_dim(1:end-1), feature_dim]);
                
                % reshape samples x gen dim into one dimension
                new_sz_search = size(Xtest);
                Xtest = reshape(Xtest, [new_sz_search(1)*new_sz_search(2), new_sz_search(3:end)]);
            end
            if save_trainlabel, all_trainlabel{rr,kk} = trainlabel; end

            % Remember sizes
            sz_Xtrain = size(Xtrain);
            sz_Xtest = size(Xtest);
            
            for ix = dim_loop                       % ---- search dimensions ----
                                
                % Training data for current search position
                if has_neighbours && ~cfg.append
                    % --- searchlight --- define neighbours for current iteration
                    ix_nb = cellfun( @(N,f) find(N(f,:)), cfg.neighbours, ix, 'Un',0);
                    % train data
                    X_ix = Xtrain(sample_skip{:}, ix_nb{:}, feature_skip{:});
                    X_ix = reshape(X_ix, [sz_Xtrain(sample_dim), prod(cellfun(@numel, ix_nb)) * nfeat]);
                    % test data
                    Xtest_ix = squeeze1(Xtest(sample_skip{:}, ix_nb{:}, feature_skip{:}));
                    Xtest_ix = reshape(Xtest_ix, [sz_Xtest(sample_dim), prod(cellfun(@numel, ix_nb)) * nfeat]);
                elseif cfg.append
                    % search dimensions are appended to train data
                    X_ix = Xtrain;
                    Xtest_ix = Xtest;
                else
                    if isempty(gen_dim),    ix_test = ix;
                    else,                   ix_test = ix(1:end-1);
                    end
                    X_ix = squeeze(Xtrain(sample_skip{:}, ix{:}, feature_skip{:}));
                    Xtest_ix = squeeze1(Xtest(sample_skip{:}, ix_test{:}, feature_skip{:}));
                end
                
                % Train classifier
                cf= train_fun(cfg.hyperparameter, X_ix, trainlabel);

                % Obtain classifier output (labels, dvals or probabilities)
                if isempty(gen_dim)
                    cf_output{rr,kk,ix{:}} = mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest_ix);
                else
                    % generalization: we have to reshape classifier output back
                    cf_output{rr,kk,ix{:}} = reshape( mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest_ix), numel(testlabel{rr,kk}),[]);
                end
                if save_model, all_model{rr,kk,ix{:}} = cf; end
            end

        end
        if cfg.feedback, fprintf('\n'), end
    end

    % Average classification performance across repeats and test folds
    avdim= [1,2];

elseif has_second_dataset
    % -------------------------------------------------------
    % Transfer classification (aka cross decoding) using two datasets. The 
    % first dataset acts as train data, the second as test data.
    
    % Initialize classifier outputs
    if cfg.append
        cf_output = cell([1, 1]);
    else
        cf_output = cell([1, 1, sz_search]);
    end
    if save_model, all_model = cell(size(cf_output)); end

    % Preprocess train data
    [tmp_cfg, X, clabel] = mv_preprocess(cfg, X, clabel);
    
    % Preprocess test data
    [~, X2, clabel2] = mv_preprocess(tmp_cfg, X2, clabel2);
    
    Xtrain = X;
    Xtest = X2;
    trainlabel = clabel;
    testlabel = clabel2;
    
    if ~isempty(gen_dim)
        Xtest = permute(Xtest, [sample_dim, search_dim(end), search_dim(1:end-1), feature_dim]);
        % reshape samples x gen dim into one dimension
        new_sz_search = size(Xtest);
        Xtest = reshape(Xtest, [new_sz_search(1)*new_sz_search(2), new_sz_search(3:end)]);
    end
    
    % Remember sizes
    sz_Xtrain = size(Xtrain);
    sz_Xtest = size(Xtest);
    
    for ix = dim_loop                       % ---- search dimensions ----
        
        % Training data for current search position
        if has_neighbours && ~cfg.append
            ix_nb = cellfun( @(N,f) find(N(f,:)), cfg.neighbours, ix, 'Un',0);
            X_ix = Xtrain(sample_skip{:}, ix_nb{:}, feature_skip{:});
            X_ix = reshape(X_ix, [sz_Xtrain(sample_dim), prod(cellfun(@numel, ix_nb)) * nfeat]);
            Xtest_ix = squeeze1(Xtest(sample_skip{:}, ix_nb{:}, feature_skip{:}));
            Xtest_ix = reshape(Xtest_ix, [sz_Xtest(sample_dim), prod(cellfun(@numel, ix_nb)) * nfeat]);
        elseif cfg.append
            % search dimensions are appended to train data
            X_ix = Xtrain;
            Xtest_ix = Xtest;
        else
            if isempty(gen_dim),    ix_test = ix;
            else,                   ix_test = ix(1:end-1);
            end
            X_ix = squeeze(Xtrain(sample_skip{:}, ix{:}, feature_skip{:}));
            Xtest_ix = squeeze1(Xtest(sample_skip{:}, ix_test{:}, feature_skip{:}));
        end
        
        % Train classifier
        cf= train_fun(cfg.hyperparameter, X_ix, trainlabel);
        
        % Obtain classifier output (labels, dvals or probabilities)
        if isempty(gen_dim)
            cf_output{1,1,ix{:}} = mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest_ix);
        else
            cf_output{1,1,ix{:}} = reshape( mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest_ix), numel(testlabel),[]);
        end
        if save_model, all_model{ix{:}} = cf; end
    end
    
    avdim = [];
    all_trainlabel = trainlabel;
    
elseif strcmp(cfg.cv,'none')
    % -------------------------------------------------------
    % No cross-validation, just train and test once for each
    % training/testing time. This gives the classification performance for
    % the training set, but it may lead to overfitting and thus to an
    % artifically inflated performance.
    
    % Preprocess train/test data
    if ~isempty(cfg.preprocess)
        [~, X, clabel] = mv_preprocess(cfg, X, clabel);
    end
    
    % Initialize classifier outputs
    if cfg.append
        cf_output = cell([1, 1]);
    else
        cf_output = cell([1, 1, sz_search]);
    end
    if save_model, all_model = cell(size(cf_output)); end

    if ~isempty(gen_dim)
        Xtest= permute(X, [sample_dim, search_dim(end), search_dim(1:end-1), feature_dim]);
        
        % reshape samples x gen dim into one dimension
        sz_search = size(Xtest);
        Xtest= reshape(Xtest, [sz_search(1)*sz_search(2), sz_search(3:end)]);
    else
        Xtest = X;
    end
    
    % Remember sizes
    sz_Xtrain = size(X);
    sz_Xtest = size(Xtest);
    
    for ix = dim_loop                       % ---- search dimensions ----
        
        % Training data for current search position
        if has_neighbours
            % --- searchlight --- define neighbours for current iteration
            ix_nb = cellfun( @(N,f) find(N(f,:)), cfg.neighbours, ix, 'Un',0);
            % train data
            X_ix = X(sample_skip{:}, ix_nb{:}, feature_skip{:});
            X_ix= reshape(X_ix, [sz_Xtrain(sample_dim), prod(cellfun(@numel, ix_nb)) * nfeat]);
            % test data
            Xtest_ix = squeeze(Xtest(sample_skip{:}, ix_nb{:}, feature_skip{:}));
            Xtest_ix = reshape(Xtest_ix, [sz_Xtest(sample_dim), prod(cellfun(@numel, ix_nb)) * nfeat]);
        elseif cfg.append
            % search dimensions are appended to train data
            X_ix = X;
            Xtest_ix = Xtest;
        else
            if isempty(gen_dim),    ix_test = ix;
            else,                   ix_test = ix(1:end-1);
            end
            X_ix= squeeze(X(sample_skip{:}, ix{:}, feature_skip{:}));
            Xtest_ix = squeeze(Xtest(sample_skip{:}, ix_test{:}, feature_skip{:}));
        end
        
        % Train classifier
        cf= train_fun(cfg.hyperparameter, X_ix, clabel);
        
        % Obtain classifier output (labels, dvals or probabilities)
        if isempty(gen_dim)
            cf_output{1,1,ix{:}} = mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest_ix);
        else
            % we have to reshape classifier output back
            cf_output{1,1,ix{:}} = reshape( mv_get_classifier_output(cfg.output_type, cf, test_fun, Xtest_ix), numel(clabel),[]);
        end
        if save_model, all_model{ix{:}} = cf; end
    end

    all_trainlabel = clabel;
    testlabel = clabel;
    avdim = [];
end

%% Calculate performance metrics
if cfg.feedback, fprintf('Calculating performance metrics... '), end
perf = cell(n_metrics, 1);
perf_std = cell(n_metrics, 1);
perf_dimension_names = cell(n_metrics, 1);
for mm=1:n_metrics
    if strcmp(cfg.metric{mm},'none')
        perf{mm} = cf_output;
        perf_std{mm} = [];
        perf_dimension_names{mm} = [{'repetition'} {'fold'} cfg.dimension_names(search_dim)];
    else
        [perf{mm}, perf_std{mm}] = mv_calculate_performance(cfg.metric{mm}, cfg.output_type, cf_output, testlabel, avdim);
        % performance dimension names
        if isvector(perf{mm})
            if numel(perf{mm}) == 1
                perf_dimension_names{mm} = 'metric';
            else
                perf_dimension_names{mm} = cfg.dimension_names(search_dim);
            end
        else
            if ~isempty(gen_dim)
                ix_gen_in_search_dim = find(search_dim == gen_dim);
                names = cfg.dimension_names(search_dim);
                names{ix_gen_in_search_dim} = ['train ' names{ix_gen_in_search_dim}];
                perf_dimension_names{mm} = [names repmat({'metric'}, 1, ndims(perf{mm})-numel(search_dim)-numel(gen_dim)) {['test ' cfg.dimension_names{gen_dim}]}];
            else
                perf_dimension_names{mm} = [cfg.dimension_names(search_dim) repmat({'metric'}, 1, ndims(perf{mm})-numel(search_dim)-numel(gen_dim)) cfg.dimension_names(gen_dim)];
            end
        end
    end
end
if cfg.feedback, fprintf('finished\n'), end

if n_metrics==1
    perf = perf{1};
    perf_std = perf_std{1};
    perf_dimension_names = perf_dimension_names{1};
    cfg.metric = cfg.metric{1};
end

result = [];
if nargout>1
   result.function              = mfilename;
   result.task                  = 'classification';
   result.perf                  = perf;
   result.perf_std              = perf_std;
   result.perf_dimension_names  = perf_dimension_names;
   result.testlabel             = testlabel;
   result.metric                = cfg.metric;
   result.n                     = size(X, 1);
   result.n_metrics             = n_metrics;
   result.n_classes             = n_classes;
   result.classifier            = cfg.classifier;
   result.cfg                   = cfg;
   if save_trainlabel, result.trainlabel = all_trainlabel; end
   if save_model, result.model_param = all_model; end
end