function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(size(X,1),1),X];
% calculate hidden layer neuron activation
a2 = sigmoid(a1 * Theta1');
% add ones to a2
a2 = [ones(size(a2,1),1),a2];
% calculate output layer activation, in our case it is hypothesis
a3 = sigmoid(a2 * Theta2');
hyp = a3;

% convert y to matrix of binariy values
y_binary_matrix = zeros(m,num_labels);
for i=1:m,
	tmp = zeros(1, num_labels);
	tmp(y(i)) = 1;
	y_binary_matrix(i,:) = tmp;
end

% calculate costFunction J without regularization

J = (1/m) .* sum(sum((-y_binary_matrix .* log(hyp)) - ((1-y_binary_matrix) .* log(1-hyp))));

% regularization (removing the bias columns)
newTheta1 = Theta1;
newTheta1(:,1) = zeros(size(newTheta1, 1), 1);
sum1 = sum(newTheta1(2:end) .^ 2);

newTheta2 = Theta2;
newTheta2(:,1) = zeros(size(newTheta2, 1), 1);
sum2 = sum(newTheta2(2:end) .^ 2);

reg = (lambda / (2*m)) * (sum1 + sum2);

% regularized cost function
J = J + reg;

for i = 1:m

    % neuron activations
    a1 = X(i, :);
    a1 = [ones(size(a1, 1), 1), a1];

    z2 = a1 * Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(size(a2, 1), 1), a2];

    z3 = a2 * Theta2';
    a3 = sigmoid(z3);    
    
    % delta(output layer)
    d3 = a3 - y_binary_matrix(i, :);
    d3 = d3'; % make it 10 x 1    
    
    % delta(hidden layer)
    d2_tmp = Theta2' * d3;
    d2_tmp = d2_tmp(2:end, :); % remove the bias
    d2 = d2_tmp .* sigmoidGradient(z2)';
        
    % gradient accumulation
    Theta1_grad = Theta1_grad + (d2 * a1);
    Theta2_grad = Theta2_grad + (d3 * a2);
end

% Step 5 - divide by m
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

grad_reg1 = (lambda/m) .* newTheta1;
grad_reg2 = (lambda/m) .* newTheta2;

Theta1_grad = Theta1_grad + grad_reg1;
Theta2_grad = Theta2_grad + grad_reg2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
