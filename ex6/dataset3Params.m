function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;

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
lowest_error=100000;
Co=[0.01,0.03,0.1,0.3,1,3,10,30];
sigmao = [0.01,0.03,0.1,0.3,1,3,10,30];
for i=1:size(Co,2),
    for j=1:size(sigmao,2),
        theta=svmTrain(X,y,Co(i),@(x1,x2)gaussianKernel(x1,x2,sigmao(j)));
        predictions=svmPredict(theta,Xval);
        error=mean(double(predictions~=yval));
        if error<=lowest_error,
            lowest_error=error;
            bestC=Co(i);
            bestsigma=sigmao(j);
        end;
    end;
end;
C=bestC;
sigma=bestsigma;
        
        


% =========================================================================

end
