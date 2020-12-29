% UNDERSTANDING METRICS
%
% The purpose of this tutorial is to gain a better understanding of
% the different classification and regression metrics in MVPA-Light. It 
% covers the following topics:
%
% (1) Relationship between dvals (decision values), accuracy, and raw
%     classifier output
% (2) Looking at three types of raw classifier output: 
%     class labels, dvals, and probabilities
% (x) Relationship between dvals and AUC
% (x) Regression: relationship between MAE and MSE
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
% Note: If you are new to working with MVPA-Light, make sure that you
% complete the introductory tutorials first:
% - getting_started_with_classification
% - getting_started_with_regression
% They are found in the same folder as this tutorial.

close all
clear

%% Loading example data
[dat, clabel] = load_example_data('epoched3');
X = dat.trial;

%% (1) Relationship between dvals (decision values), accuracy, and raw classifier output

% To keep the results simple, we will use just 1 hold out set as
% cross-validation approach. We will then extract classification accuracy
% ('accuracy'), decision values ('dval'), and raw classifier output ('none').
% Note that the raw classifier output can be single-trial predicted class
% labels, or decision values, or probabilities. 
% We will use a LDA classifier which, by default, produces decision values.
cfg = [];
cfg.metric      = 'none'; % {'accuracy' 'dval' 'none'};
cfg.cv          = 'holdout';
cfg.p           = 0.5;
cfg.repeat      = 1;
cfg.classifier  = 'lda';
    

[perf, result] = mv_classify_across_time(cfg, X, clabel);

% Before looking at the results let us recall the dimensions of the data
% which are [313, 30, 131], that is 313 samples, 30 channels, 131 time points 
size(X)

% Let's look at perf now:
% Recall perf is a cell array with three elements (ie length(perf)=3)
% because we requested 3 metrics. So perf{1} corresponds to 'accuracy',
% perf{2} corrresponds to 'dval', and perf{3} to 'none' (raw classifier outputs). 
% If we print the cell array we can have a closer look at the dimensions
perf

% Let us focus at perf{1} and perf{2} for now. We see that size(perf{1}) is
% is [131,1]
size(perf{1})

% whereas size(perf{2}) is [131,2]. 
size(perf{2}) 

% So for 'accuracy' we a vector of accuracy values, one accuracy for each
% time point. However, for 'dval' we get two such vectors. Let us visualize
% the result to see why
mv_plot_result(result)

%%% TODO: fix visualisatoin for NONE data


dval = perf{2}

clf
plot(dval(:,1)) % first vector 
hold all
plot(dval(:,2)) % second vector
legend({'class 1' 'class 2'})



%%%%%% EXERCISE 1 %%%%%%
% Looking at the dimensions of perf for the 'none' metric, we get 
% size(perf{3}) = [1, 1, 131]? Why is it not [131, 1] like the accuracy
% metric? What do the first two dimensions encode?
% Hint: rerun the analysis with 3-fold cross-validation and 2 repetitions
% and look at the size again.
%%%%%%%%%%%%%%%%%%%%%%%%

%% 


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
cfg = [];
cfg.metric      = {'accuracy' 'dval' 'none'};
cfg.cv          = 'kfold';
cfg.k           = 3;
cfg.repeat      = 2;
    
[perf, result] = mv_classify_across_time(cfg, X, clabel);

perf

% Now we see that size(perf{3}) = [2, 3, 131]. So the first dimension
% represents the number of repetitions (2), the second the number of test
% folds (3), and the third is the number of time points. For metric='none'
% we get separate results for each repetition and fold because the results
% are not averaged across folds/repetitions.
