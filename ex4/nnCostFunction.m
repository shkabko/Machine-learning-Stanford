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

Yk=[1:num_labels];
         
% You need to return the following variables correctly 
J = 0;
J1 = 0;
J2 = 0;
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

%A1=[ones(m,1) X]; % [500,401] one added
%Z2=A1*Theta1'; % [500,25] size
%A2=sigmoid(Z2); % [500,25] size
%m1 = size(A2, 1);
%A2=[ones(m1,1) A2]; %[500, 26] one added
%Z3=A2*Theta2'; %[500,10] row vector
%A3=sigmoid(Z3); %output vector 5000 by 10



X=[ones(m,1) X];   %size [5000,400+1] before
A2=sigmoid(X*Theta1');
m1 = size(A2, 1);
A2 = [ones(m1, 1) A2];
A3=sigmoid(A2*Theta2'); %size [5000,10]

    %[xx,p] = max(A3,[],2); prediction of maxim. values xx with labels in the "2" rows[1..10]
           
           I=eye(num_labels)(y,:); % fills matrix

           for i=1:m
           for j=1:num_labels
        
           J = J + (1/m)*(-I(i,j).*log(A3(i,j))-log(1-A3(i,j))+I(i,j).*log(1-A3(i,j)));
          
%nugrad = grad+(1/m)*(sigmoid(theta'*X(i,:)')-y(i))*X(i,:)';

end
end
           
           J1=(lambda/(2*m))*sum(sum(Theta1(:,2:end).^2));
           
           %for j=1:hidden_layer_size;
           %for k=2:input_layer_size+1;
           %J1=J1+(lambda/(2*m))*Theta1(j,k).^2;
           %end;
           %end;
           
           J2=(lambda/(2*m))*sum(sum(Theta2(:,2:end).^2));
           
           %for j=1:num_labels;
           %for k=2:hidden_layer_size+1;
           %J2=J2+(lambda/(2*m))*Theta2(j,k).^2;
           %end;
           %end;
           
           J=J+J1+J2;

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
           
           
           %A1=[ones(m,1) X]; % [5000,401] one added
        
           Z2=X*Theta1'; % [5000,25] size
           A2=sigmoid(Z2); % [5000,25] size
           A2=[ones(m,1) A2]; %[5000, 26] one added
           Z3=A2*Theta2'; %[5000,10] row vector
           A3=sigmoid(Z3); %output vector 5000 by 10
           
           Y=eye(num_labels)(y,:); % vector 5000 by 10
           
           
           d3=A3-Y;  %calculate d3 vector 5000 by 10
           d2=d3*Theta2(:,2:end).*(sigmoidGradient(Z2));  %calculate d2 size 5000*25
          
           Theta1_grad=(1/m)*d2'*X; % 25by401
           Theta2_grad=(1/m)*d3'*A2; % 10by26
%          D=(1/m)*(Theta1_grad+Theta2_grad); total accumulated gradient
           
           
           
           
           
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
           Theta1(:,1)=0;
           Theta2(:,1)=0;
           Theta1_grad=Theta1_grad+(lambda/m)*Theta1;
           Theta2_grad=Theta2_grad+(lambda/m)*Theta2;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
