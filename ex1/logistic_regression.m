function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));


  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
%%% YOUR CODE HERE %%%

% f = -?i(y(i)log(h?(x(i)))+(1-y(i))log(1-h?(x(i))))
% g = ?ix(i)j(h?(x(i))-y(i))

h_theta = sigmoid(theta' * X); % compute h_theta ahead to save time
for i = 1:m
  %h_theta = 1 / (1 + exp(-theta' * X(:,i)));
  f = f + ((y(i) * log(h_theta(i))) + ((1 - y(i)) * log(1 - h_theta(i))));
  g = g + X(:,i) * (h_theta(i) - y(i));
end
f = -f;