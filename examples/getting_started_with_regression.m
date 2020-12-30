% GETTING STARTED WITH REGRESSION
% 
% This is the go-to tutorial if you are new to the toolbox and want to
% get started with regression. It covers the following topics:
%
% (1) Loading example data
% (2) Cross-validation and explanation of the cfg struct
% (3) Classification of data with a time dimension
% (4) Time generalization (time x time classification)
% (5) Plotting results
%
% It is recommended that you work through this tutorial step by step. To
% this end, copy the line of code that you are currently reading and paste
% it into the Matlab console. 
%
% There will be exercises throughout the tutorial. Try to do the exercise,
% you can then check your code against the solution at the end of this
% file.
%
% Note: we will use machine learning terminology such as cross-validation,
% features, classifier, train and test set. If you are unfamiliar with
% these terms, please check out the glossary in section 1.1 of the
% MVPA-Light paper (www.frontiersin.org/articles/10.3389/fnins.2020.00289)
% and read the tutorial papers that are mentioned there.
%
% Troubleshooting: If the code crashes on your computer, make sure that you
% always have the latest version of the toolbox and that you are using
% Matlab version 2012 or newer.
%
% Documentation:
% The Github Readme file is the most up-to-date documentation of the
% toolbox. You will find an explanation of the function, classifiers,
% metrics and parameters there: github.com/treder/MVPA-Light/blob/master/README.md
%
% Next steps: Once you finished this tutorial, you can continue with one
% of the other example scripts:
% - understanding_preprocessing: how to use the cfg.preprocess field for
%   nested preprocessing
% - understanding_train_and_test_functions: how to train and test models
%   directly, without using the high-level interface (e.g. mv_classify)

close all
clear

%% (1) Loading example data
% MVPA-Light comes with example datasets to get you started right away. 
% The dataset is taken from this study: 
% https://iopscience.iop.org/article/10.1088/1741-2560/11/2/026009/meta
% The data has been published at
% http://bnci-horizon-2020.eu/database/data-sets (dataset #15). Out of this
% data, 
%
% Here, it will be explained how to load the data and how to use it.
% MVPA-Light has a custom function, load_example_data, which is exclusively
% used for loading the example datasets shipped with the toolbox. Let's
% load one of the datasets:

% Load data (which is located in the MVPA-Light/examples folder)
[dat, clabel] = load_example_data('epoched3');

% We loaded a dataset called 'epoched3' and it returned two variables, dat
% and clabel. Now type dat in the console:
dat

% dat is actually a FieldTrip structure, but we are only interested in the
% data contained in the dat.trial field for now. Let us assign this data to 
% the variable X and then look at the size of X:
X = dat.trial;
size(X)

% It has the dimensions 313 x 30 x 131. The other variable is the class
% labels which can be used for classification. In this tutorial, we are
% interested in regression and therefore we will ignore this variable
% (check out getting_started_with_classification to learn more about clabel
% and how it's used in classification). For now, we will ignore the fact
% that there is different classes are we will frame a simple regression
% problem in the next section.


%%
%%% In this example we look at regression tasks. We will use the same
%%% datasets used in the classification examples. Since no response
%%% variable (e.g. reaction time) was recorded we will use the number of
%%% the trial (as a proxy for time in the experiment) as response variable.
%%% This can tell us whether there are changes in amplitude across the
%%% duration of the experiment. 

close all
clear all

% Load data (in /examples folder)
[dat,clabel,chans] = load_example_data('epoched1');

% keep only class 2
dat.trial = dat.trial(clabel==2, :, :);

% the response variable is simply the number of the trial
y = [1:size(dat.trial, 1)]';

% [dat2,clabel2] = load_example_data('epoched2');
% [dat3,clabel3] = load_example_data('epoched3');

% Plot data
close
h1= plot(dat.time, squeeze(mean(dat.trial, 1)), 'r'); hold on
grid on
xlabel('Time [s]'),ylabel('EEG amplitude')
title('ERP')

%% Regression on the [0.2 - 0.3] s ERP component

% mark the start and end of component in the plot
yl = ylim;
plot([0.2, 0.2], yl, 'k--'); 
plot([0.3, 0.3], yl, 'k--');

time_points = find( (dat.time >= 0.2) & (dat.time <= 0.3) );

X = squeeze(mean(dat.trial(:, :, time_points), 3));

% Set up the structure with options for mv_regress
cfg = [];
cfg.model   = 'ridge';               % ridge regression (inludes linear regression)
cfg.hyperparameter.lambda = [ 0, 1, 2, 10];
cfg.metric  = {'mse', 'mean_absolute_error'}; % can be abbreviated as 'mae'
cfg.dimension_names = {'samples' 'channels'};

% Call mv_regress to perform the regression 
[perf, result] = mv_regress(cfg, X, y);

mv_plot_result(result)

%% Regression across time
% Perform the same regression but this time for every time point, yielding
% MAE as a function of time

% Set up the structure with options for mv_regress
cfg = [];
cfg.model   = 'ridge';
cfg.metric  = 'mae';                 % = mean absolute error
cfg.dimension_names = {'samples' 'channels', 'time points'};

[perf, result] = mv_regress(cfg, dat.trial, y);

% ax = mv_plot_1D(dat.time, perf, result.perf_std, 'ylabel', cfg.metric)
mv_plot_result(result, dat.time)


%% Compare ridge regression / kernel ridge / Support Vector Regression
% To illustrate how kernels tackle non-linear problems, we will
% here create an 1-dimensional non-linear dataset. We will then train ridge
% regression, kernel ridge, and Support Vector Regression (SVR) models and
% compare them. 
% Note: The SVR model requires an installation of LIBSVM, see
% train_libsvm.m for details

x = linspace(0, 12, 100)';
y = -.1*x.^2 + 3*sin(x);     % SINUSOID WITH QUADRATIC TREND
% y = 2*mod(x, 3) + 0.4 * x; % SAWTOOTH FUNCTION
y_plus_noise  = y + randn(length(y), 1);

close all
plot(x,y, 'r', 'LineWidth', 2)
hold on
plot(x,y_plus_noise, 'ko')
legend({'Signal' 'Signal plus noise'})
title('True signal and data')

% Train ridge model and get predicted values 
param = mv_get_hyperparameter('ridge');
model = train_ridge(param, x, y);
y_ridge = test_ridge(model, x);

% Train kernel ridge model and get predicted values 
param = mv_get_hyperparameter('kernel_ridge');
% param.kernel = 'polynomial';
model = train_kernel_ridge(param, x, y);
y_kernel_ridge = test_kernel_ridge(model, x);

% Train SVR model and get predicted values.
% We will use the LIBSVM toolbox here, which supports both 
% classification (SVM) and regression (SVR).
param = mv_get_hyperparameter('libsvm');

% Set svm_type to 3 for support vector regression
param.svm_type = 3; 
model = train_libsvm(param, x, y);
y_svr = test_libsvm(model, x);

figure,hold on
% plot(x,y, 'r', 'LineWidth', 2)  % true signal
plot(x,y_plus_noise, 'ko')
plot(x, y_ridge, 'b')   % ridge prediction
plot(x, y_kernel_ridge, 'k')   % kernel ridge prediction
plot(x, y_svr, 'g')   % SVR prediction

legend({'Data' 'Ridge regression' 'Kernel ridge' 'SVR'})
title('Predictions')






%% SOLUTIONS TO THE EXERCISES
%% SOLUTION TO EXERCISE 1
% We only need to find the index of channel Fz, the rest is the same
ix = find(ismember(dat.label, 'Fz'));
figure
plot(dat.time, ERP_attended(ix, :))
hold all, grid on
plot(dat.time, ERP_unattended(ix, :))

legend({'Attended' 'Unattended'})
title('ERP at channel Fz')
xlabel('Time [s]'), ylabel('Amplitude [muV]')

%% SOLUTION TO EXERCISE 2
% Looking in the Readme file, we can see that the Naive Bayes classifier
% is denoted as naive_bayes 
cfg = [];
cfg.classifier  = 'naive_bayes';

perf = mv_classify(cfg, X, clabel);

%% SOLUTION TO EXERCISE 3
% Looking at the Readme file, we can see the precision and recall are
% simply denoted as 'precision' and 'recall'
cfg = [];
cfg.metric      = {'precision', 'recall'};
perf = mv_classify(cfg, X, clabel);
fprintf('Precision = %0.2f, Recall = %0.2f\n', perf{:})

%% SOLUTION TO EXERCISE 4
% Leave-one-out cross-validation is called 'leaveout' and we need to set
% the cv field to define it. Now, you will see that we have 313 folds and
% only one repetition: In leave-one-out cross-validation, each of the 313
% samples is held out once (giving us 313 folds). It does not make sense to
% repeat the cross-validation more than once, since there is no randomness
% in assigning samples to test folds any more (every sample is in a test
% fold once).
cfg = [];
cfg.cv      = 'leaveout';
perf = mv_classify(cfg, X, clabel);

%% SOLUTION TO EXERCISE 5
% Cross-validation relies on random assignments of samples into folds. This
% randomness leads to some variability in the outcome. For instance, let's
% assume you find a AUC of 0.78. When you rerun the analysis it changes to
% 0.75 or 0.82. Having multiple repetitions and averaging across them
% stabilizes the estimate.
%
% We can check this empirically by first running a classification analysis
% several times with 1 repeat and then comparing it to 5 repeats. For the 5
% repeats, the variability (=standard deviation) of the classification
% scores should be smaller than for the 1 repeat case, illustrating that
% it is more stable/replicable.

one_repeat = zeros(10,1);
five_repeats = zeros(10,1);

% for simplicity and speed, we reduce the data to 2D 
X = dat.trial(:,:,floor(end/2));

% one repeat
cfg = [];
cfg.repeat      = 1;
cfg.feedback    = 0;  % suppress output
for ii=1:10
    one_repeat(ii) = mv_classify(cfg, X, clabel);
end

% five repeats
cfg = [];
cfg.repeat      = 5;
cfg.feedback    = 0;  % suppress output
for ii=1:10
    five_repeats(ii) = mv_classify(cfg, X, clabel);
end

one_repeat'
five_repeats'

fprintf('Std for one repeat: %0.5f\n', std(one_repeat))
fprintf('Std for five repeats: %0.5f\n', std(five_repeats))

%% SOLUTION TO EXERCISE 6
% We simply set cfg.classifier, cfg.metric, cfg.k and cfg.repeat to the
% required values and then perform the classification.
X = dat.trial;

cfg = [];
cfg.classifier      = 'logreg';
cfg.metric          = 'kappa';
cfg.k               = 20;
cfg.repeat          = 1;
perf = mv_classify_across_time(cfg, X, clabel);

close all
plot(dat.time, perf, 'o-')
grid on
xlabel('Time'), ylabel(cfg.metric)
title('Classification across time')

%% SOLUTION TO EXERCISE 7
cfg = [];
cfg.cv          = 'none';
cfg.metric      = 'auc';
auc = mv_classify_timextime(cfg, dat.trial, clabel);

% when cross-validation is turned off, most of the pattern is very
% similar. However, a diagonal appears running from the bottomn left to the 
% top right. This is because, without cross-validation, we get some
% overfitting (this is why we have good classification performance even in 
% the pre-stimulus phase). Cross-validation prevents this from happening.
figure
imagesc(dat.time, dat.time, auc)
set(gca, 'YDir', 'normal')
colorbar
grid on
ylabel('Training time'), xlabel('Test time')
title('No cross-validation')

%% SOLUTION TO EXERCISE 8
% We just need to reverse the order of the arguments, feeding in dat2 and
% clabel2 first and then dat1 and clabel1. The result is not the same, but
% this is not to be expected: in cross-decoding we identify discriminative
% patterns in the first dataset and then look for them in the second
% dataset. The discriminative patterns for two subjects are likely
% different.
cfg = [];
cfg.classifier = 'lda';
cfg.metric     = 'auc';

acc = mv_classify_timextime(cfg, dat2.trial, clabel2, dat1.trial, clabel1);

figure
imagesc(dat2.time, dat1.time, acc)
set(gca, 'YDir', 'normal')
colorbar
grid on
ylabel('Participant 2 time'), xlabel('Participant 1 time')

%% SOLUTION TO EXERCISE 9
% With multiple metrics, the mv_plot_result function simply creates
% multiple figures, one for each metric. Note that the name of the metric
% appears on the y-axis or on top of the colorbar.
X = mean(dat.trial(:,:,100), 3);

cfg = [];
cfg.metric = {'precision', 'recall'};
[perf, result] = mv_classify(cfg, X, clabel);
mv_plot_result(result)

cfg = [];
cfg.metric = {'precision', 'recall'};
[perf, result] = mv_classify_across_time(cfg, dat.trial, clabel);
mv_plot_result(result, dat.time)

cfg = [];
cfg.metric = {'precision', 'recall'};
[perf, result] = mv_classify_timextime(cfg, dat.trial, clabel);
mv_plot_result(result, dat.time, dat.time);
