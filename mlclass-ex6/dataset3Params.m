function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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

	maxError = Inf;
	init_C_values = [0.01; 0.03; 0.1; 0.3;1; 3; 10; 30];
	init_sigma_values = [0.01; 0.03; 0.1; 0.3;1; 3; 10; 30];

	for i=1:length(init_C_values),
		for j=1:length(init_sigma_values),
			% Train the classifier to get the model
			svmModel = svmTrain( X, y, init_C_values(i), @(x1, x2) gaussianKernel( x1, x2, init_sigma_values(j) ) );
			% Apply model over cross validation examples
			svmPredictions = svmPredict(svmModel, Xval);
			% Verify the predictions on cross validation set
			predError = mean(double(svmPredictions ~= yval));
			%take the min error out of all iterations and update C and sigma accordingly
			if(predError < maxError),
				maxError = predError;
				C = init_C_values(i);
				sigma = init_sigma_values(j);
			end
		end
	end

% =========================================================================

end
