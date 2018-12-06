function [J, grad, d_W1, d_W2] = nnCostFunction(nn_params, ...
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


Y = zeros(num_labels,m);
for sample = 1:m,
    VecY = Y(:,sample);
    VecY(y(sample)) =1;
    Y(:,sample) = VecY;
end
    

X_aug = [ones(m,1) , X]; % samples by pixels 

Hidden = sigmoid(Theta1 * X_aug'); % nodes by samples

Hid_aug = [ones(1,m); Hidden]; % nodes + 1 by samples

Out = sigmoid(Theta2 * Hid_aug); % classes by samples

% sum on classes, then by samples
J = sum(sum( (-Y).*log(Out) - (1-Y).*log(1-Out), 2))/m;

penalty = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2));
% total 
J = J + lambda * penalty /(2*m);


% If we do it sample by sample 
% for each_sample = 1:m,
%     % Forward Prop
%     sample_x = X(m,:);
%     sample_x = [1,sample_x]';
%     Hidden = sigmoid(Theta1 * sample_x);
%     Hidden = [1; Hidden];
%     Out = sigmoid(Theta2 * Hidden);
%     % label in scheme
%     Vec_y = zeros(num_labels,1);
%     Vec_y (y(each_sample)) =1;
%     J_sample(each_sample) = sum( - Vec_y. * log( Out ) - (1-Vec_y). * log( 1-Out ) );
% end

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



% Error back prop
% For Theta2:
err3 = Out -Y; % classes by samples
% For Theta1:
z2 = Theta1 * X_aug';
z2 = [ones(1,m);z2];
err2 = (Theta2' *  err3) .* sigmoidGradient(z2); % node + 1 by samples
err2 (1,:) = []; % node by samples
% There is NO err1.


% delta * a '
% The matrix multiplication sums over samples
% Which does the accumulation of triangles implicitly. 
d_W2 = err3 * Hid_aug'; % classes by node +1
d_W1 = err2 * X_aug;  % node +1 by pixels + 1
%d_W1(1,:) = []; % node by pixels + 1

% This is the gradient!
d_Theta1 = d_W1/m;
d_Theta2 = d_W2/m;


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%





Penalty_Theta1 = lambda * Theta1 /m;
% remember to remove penalty for bias
Penalty_Theta1(:,1) = zeros(size(Theta1,1),1);

Penalty_Theta2 = lambda * Theta2 /m;
Penalty_Theta2(:,1) = zeros(size(Theta2,1),1);


% together
Theta1_grad = d_Theta1 + Penalty_Theta1;
Theta2_grad = d_Theta2 + Penalty_Theta2;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
