function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% from the hw we got a list of candidate C
possible =[0.01 0.03 0.1 0.3 1 3 10 30];
num_try = length(possible);

allPerf = zeros(num_try);

for C_try = 1:num_try,
    for Sigma_try = 1:num_try, 
        model  = svmTrain(X, y, possible(C_try), ...
            @(x1, x2) gaussianKernel(x1, x2, possible(Sigma_try) ) );
        predictions = svmPredict(model, Xval);
        allPerf(C_try,Sigma_try) = mean(double(predictions ~= yval));
    end
end

% locate min error perf
[~,index] = min(allPerf);

C = possible(index(1));
sigma = possible (index(2));

% answer is:
C = 1.000000;
sigma = 0.100000;
% =========================================================================

end
