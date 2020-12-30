% GETTING STARTED WITH CLASSIFICATION
% 
% This is the go-to tutorial if you are new to the toolbox and want to
% get started with classifying data. It covers the following topics:
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
% - example_classify_multidimensional_data: how to perform more complex 
%   classification analyses using e.g. time-frequency data and searchlight
%   analysis.
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
% http://bnci-horizon-2020.eu/database/data-sets (dataset #15). 
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

% It has the dimensions 313 x 30 x 131. Now let's look at the size of the
% other variable, clabel, and let's also print out its unique values
size(clabel)
unique(clabel)

% So clabel is a vector of size 313 x 1 and it contains only 1's and 2's. This
% number coincides with the first dimension of X, and it turns out that 313
% is the number of trials (called 'samples' in MVPA-Light) in the dataset.
% For each trial, clabel tells us which class the trial belongs to. 
% This dataset comes from an auditory oddball paradigm, and class 1 refers
% to trials wherein participants were presented a sound there were supposed
% to attend to (attended sounds), class 2 refers to trials wherein sounds
% were presented that the participant should not attend to (unattended
% sounds). Let's look at class labels for the first 20 trials
clabel(1:20)'

% we can see that the first 12 trials are class 2 (unattended sounds) where
% as trials 13-20 are of class 1 (attended sounds). To visualize the data,
% we can calculate the ERP for each class separately. Thus, we need to
% extract the indices of trials corresponding to each class
ix_attended = (clabel==1);    % logical array for selecting all class 1 trials 
ix_unattended = (clabel==2);  % logical array for selecting all class 2 trials 

% Let us print the number of trials in each class and the select the data
% from X:
fprintf('There is %d trials in class 1 (attended).\n', sum(ix_attended))
fprintf('There is %d trials in class 2 (unattended).\n', sum(ix_unattended))

X_attended = X(ix_attended, :, :);
X_unattended = X(ix_unattended, :, :);

% We have 102 trials in class 1 and 211 trials in class 2. This should
% coincide with the first dimension in X_attended and X_unattended, let's
% double check
size(X_attended)
size(X_unattended)

% To calculate the ERP, we now calculate the mean across the trials (first
% dimension). We use the squeeze function to the reduce the array from 3D
% to 2D, since we don't need the first dimension any more.
ERP_attended = squeeze(mean(X_attended, 1));
ERP_unattended = squeeze(mean(X_unattended, 1));
fprintf('Size of ERP_attended: [%d %d]\n', size(ERP_attended))
fprintf('Size of ERP_unattended: [%d %d]\n', size(ERP_unattended))

% Both ERPs now have the same size 30 (channels) x 131 (time points). Let
% us plot the ERP for channel Cz. To find the index of this channel, we
% need to use the channel labels (dat.label). We also need to define the
% time on the x-axis and will use dat.time for this.
ix = find(ismember(dat.label, 'Cz'));
figure
plot(dat.time, ERP_attended(ix, :))
hold all, grid on
plot(dat.time, ERP_unattended(ix, :))

legend({'Attended' 'Unattended'})
title('ERP at channel Cz')
xlabel('Time [s]'), ylabel('Amplitude [muV]')

%%%%%% EXERCISE 1 %%%%%%
% Now it's your turn: 
% Create another ERP plot, but this time select channel Fz. 
%%%%%%%%%%%%%%%%%%%%%%%%

% finally, let's plot the ERP for *all* channels. The plot will be more
% busy, but remember that each line now designates a different channel
figure
plot(dat.time, ERP_attended, 'r-')
hold all, grid on
plot(dat.time, ERP_unattended, 'b-')

title('ERP at all channels (red=attended, blue=unattended)')
xlabel('Time [s]'), ylabel('Amplitude [muV]')

% From this ERPs, it looks like the two classes are well-separated in the 
% 0.6 - 0.8 sec window. Our first analysis will focus just on this time
% window. To this end, we will average the time dimension in this window
% and discard the rest of the times. 
ival = find(dat.time >= 0.6 & dat.time <= 0.8);  % find the time points corresponding to 0.6-0.8 s

% Extract the mean activity in the interval as features
X = squeeze(mean(dat.trial(:,:,ival),3));
size(X)

% Note that now we went back to a different representation of the data: X
% is now 313 (samples) x 30 (channels), and the channels will serve as our
% features. This is because classification is usually on the single-trial
% level, we only calculated the ERPs for visualization.


%% (2) Cross-validation and explanation of the cfg struct
% So far we have only loaded plotted the data. In this section, we will get
% hands on with the toolbox. We will use the output of the previous
% section, the [samples x channels] matrix X. Let's jump straight into it:
cfg = [];
perf = mv_classify(cfg, X, clabel);

% There seems to be a lot going on here, so let's unpack the questions that
% might come up:
% 1. What happened? If we read the output on the console, we can figure out
% the following: mv_classify performed a cross-validation classification
% analysis using 5-fold cross-validation (k=5), 5 repetitions, using an 
% LDA classifier. This is simply the default behaviour if we don't specify
% anything else.
% 2. What is perf? Perf refers to 'performance metric', a measure of how
% good of a job the classifier did. By default, it calculates
% classification accuracy.
fprintf('Classification accuracy: %0.2f\n', perf)
% Hence, the classifier could distinguish both classes with an accuracy of
% 78% (0.78).

% 3. What does cfg do, it was empty after all?
% cfg controls all aspects of the classification analysis: choosing the
% classifier, a metric, preprocessing and definint the cross-validation.
% For instance, let us change the classifier to Logistic Regression
% (logreg).

cfg = [];
cfg.classifier  = 'logreg';

perf = mv_classify(cfg, X, clabel);

%%%%%% EXERCISE 2 %%%%%%
% Look at the available classifiers at 
% https://github.com/treder/MVPA-Light/blob/master/README.md#classifiers
% Do the classification again, this time using a Naive Bayes classifier.
%%%%%%%%%%%%%%%%%%%%%%%%

% Now we know how to set a classifier, let's see how we can change the
% metric that we want to be calculated.  Let's go for area under the ROC
% curve (auc) instead of accuracy. We will see that the value is higher
% than that obtained for classification accuracy.
cfg = [];
cfg.metric      = 'auc';
perf = mv_classify(cfg, X, clabel);
fprintf('AUC: %0.2f\n', perf)

% We can also calculate both AUC and accuracy at the same time using a cell
% array. Now perf will be a cell array, the first value is the AUC value,
% the second value is classification accuracy.
cfg = [];
cfg.metric      = {'auc', 'accuracy'};
perf = mv_classify(cfg, X, clabel);

perf

%%%%%% EXERCISE 3 %%%%%%
% Look at the available classification metrics at 
% https://github.com/treder/MVPA-Light/blob/master/README.md#metrics
% Do the classification again, this time calculating precision and recall.
%%%%%%%%%%%%%%%%%%%%%%%%

% We know now how to define the classifier and the performance metric. We
% still need to understand how to change the cross-validation scheme. Let us
% perform k-fold cross-validation with 10 folds (i.e. 10-fold
% cross-validation) and 2 repetitions. Note how the output on the console 
% changes.
cfg = [];
cfg.k           = 10;
cfg.repeat      = 2;
perf = mv_classify(cfg, X, clabel);


%%%%%% EXERCISE 4 %%%%%%
% Look at the description of cross-validation at 
% https://github.com/treder/MVPA-Light/blob/master/README.md#cv
% Do the classification again, but instead of k-fold cross-validation use
% leave-one-out (leaveout) cross-validation.
%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% EXERCISE 5 %%%%%%
% This is a conceptual question: why is it useful to have multiple 
% repetitions of the cross-validation analysis? Why don't we just run it
% once?
%%%%%%%%%%%%%%%%%%%%%%%%

%% (3) Classification of data with a time dimension
% If the data X is three-dimensional e.g. [samples x channels x time points], 
% we can perform a classification for every time point separately. This is 
% useful e.g. for event-related experimental designs.
% Let's go back to the original data then, which had 313 samples x 30 EEG
% channels x 131 time points.
X = dat.trial;
size(X)

% We can again use mv_classify. X now has 3 dimensions, but mv_classify
% simply loops over any additional dimension. We now obtain a
% classification accuracy for each time point. When we plot it, we can see
% that classification performance is high between 0.2 - 0.8 sec.
cfg = [];
perf = mv_classify(cfg, X, clabel);

close all
plot(dat.time, perf, 'o-')
grid on
xlabel('Time'), ylabel('Accuracy')
title('Classification across time')

% For [samples x features x time] data, MVPA-Light has also a specialized
% function called mv_classify_across_time. It does the same thing as
% mv_classify in this case, but it can be faster so you are recommended to
% use it in these cases. The only obvious difference is that in the output
% the dimensions are now labeled as 'samples', 'features', and 'time
% points'. Both mv_classify and mv_classify_across_time use the same type
% of parameters for the cfg struct
cfg = [];
perf = mv_classify_across_time(cfg, X, clabel);

%%%%%% EXERCISE 6 %%%%%%
% Let's put together everything we learned so far: Use
% mv_classify_across_time with a Logistic Regression classifier and
% 20-fold cross-validation with 1 repetition. Use Cohen's kappa as a 
% classification metric. Plot the result.
%%%%%%%%%%%%%%%%%%%%%%%%

%% (4) Time generalization (time x time classification): 
% Sometimes we want to train the classifier at a given time point t1 and 
% test it at *all* time points t2 in the trial. If we repeat this for every
% combination of training and test time points, we will obtain a [time x time] 
% matrix of results. 

% We already calculated cross-validated performance above. Here, we do the
% analysis once again, this time without cross-validation.

cfg = [];
cfg.metric      = 'auc';
auc = mv_classify_timextime(cfg, dat.trial, clabel);

% plot the image
close all
imagesc(dat.time, dat.time, auc)
set(gca, 'YDir', 'normal')
colorbar
grid on
ylabel('Training time'), xlabel('Test time')

%%%%%% EXERCISE 7 %%%%%%
% Repeat the time x time classification without cross-validation. What 
% do you notice?
%%%%%%%%%%%%%%%%%%%%%%%%

% Generalization with two datasets: So far we trained and tested on the
% same dataset. However, nothing stops us from training on one dataset and
% testing on the other dataset. This can be useful e.g. in experiments with
% different experimental conditions (eg memory encoding and memory
% retrieval) where one may want to investigate whether representations in
% the first phase re-occur in the second phase. 
%
% We do not have such example data, so instead we will do cross-participant
% classification: train on the data of participant 1, test on the data of
% participant 2
[dat1, clabel1] = load_example_data('epoched1');  % participant 1
[dat2, clabel2] = load_example_data('epoched2');  % participant 2

% To perform this, we can pass the second dataset and the second class
% label vector as extra parameters to the function call. Note that no
% cross-validation is performed since the datasets are independent. It is
% useful to use AUC instead of accuracy here, because AUC is not affected
% by different in offset and scaling that the two datasets might have.
cfg =  [];
cfg.classifier = 'lda';
cfg.metric     = 'auc';

acc = mv_classify_timextime(cfg, dat1.trial, clabel1, dat2.trial, clabel2);

close all
imagesc(dat2.time, dat1.time, acc)
set(gca, 'YDir', 'normal')
colorbar
grid on
ylabel('Participant 1 time'), xlabel('Participant 2 time')

%%%%%% EXERCISE 8 %%%%%%
% Repeat the cross-classification but train on participant 1 and test on
% participant 2. Do you expect the same result?
%%%%%%%%%%%%%%%%%%%%%%%%


%% (5) Plotting results
% So far, we have plotted the results by hand using Matlab's plot
% function. For a quick and dirty visualization, MVPA-Light has a function
% called mv_plot_result. It plots the results and nicely lays out the axes
% for us. To be able to use it, we need the result struct, which is simply
% the second output argument of any classification function.

% Let's test the visualization for 2D data first 
X = mean(dat.trial(:,:,100), 3);

cfg = [];
cfg.metric = 'auc';
[perf, result] = mv_classify(cfg, X, clabel);

% now call it passing result as an input argument. We will obtain a barplot
% representing the AUC. The height of the bar is equal to the value of
% perf. The errorbar is the standard deviation across folds and
% repetitions, an heuristic marker of how variable the performance measure
% is for different test sets.
mv_plot_result(result)

% Next, let us perform classification across time. The result will be a
% time series of AUCs. The linea represents the mean (equal to the values 
% in perf), the shaded area is again the standard deviation across
% folds/repeats.
cfg = [];
cfg.metric = 'auc';
[perf, result] = mv_classify_across_time(cfg, dat.trial, clabel);

mv_plot_result(result)

% the x-axis depicts the sample number, not the real time index. To get the
% x-axis right, we can provide the correct values as an extra argument to
% the function call
mv_plot_result(result, dat.time)

% Lastly, let us try time generalization (time x time classification). For
% the resultant plot, both the x-axis and the y-axis need to be specified.
% Therefore, we pass the parameter dat.time twice.
cfg = [];
cfg.metric = 'precision';
[perf, result] = mv_classify_timextime(cfg, dat.trial, clabel);

g = mv_plot_result(result, dat.time, dat.time);

% the output argument g contains some handles to the graphical elements
% which may be convenient when customizing the layout of the plots.

%%%%%% EXERCISE 9 %%%%%%
% What happens to the plot when you request multiple metrics at once, for
% instance, precision, recall and AUC?
%%%%%%%%%%%%%%%%%%%%%%%%

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
