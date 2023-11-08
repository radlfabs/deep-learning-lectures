import numpy as np

# Simple neural network from scratch
# estimate the sum of two numbers: y = x1 + x2 and y_hat = w1*x1 + w2*x2
# implement the training as well. Use the gradient decent algorithm
# Loss function: L(y_hat, y) = 0.5*(y_hat - y)^2 = 0.5*(w1*x1 + w2*x2 - y)^2
# Generate a randonm training set 

def generate_training_set(n: int):  
    x = np.random.uniform(0, 1, (n, 2))
    y = np.sum(x, axis=1)
    return x, y


class DenseLayer:
    def __init__(self, neurons) -> None:
        self.neurons = neurons
        
    def relu(self, inputs):
        return np.maximum(0, inputs)
    
    def softmax(self, inputs):
        exp_scores = np.exp(inputs)
        return  exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def forward(self, inputs, weights, bias):
        raise NotImplementedError
    
    def backward(self, dA_curr, W_curr, Z_curr, A_prev, activation):
        raise NotImplementedError


class Network:
    def __init__(self) -> None:
        self.layers = []
        self.architecture = []
        self.parameters = []
        self.memory = []
        self.gradients = []

    def add(self, layer):
        self.layers.append(layer)
        
    def _compile(self, data):
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                self.architecture.append({"input_dim": data.shape[1], 
                                          "output_dim": layer.neurons,
                                          "activation": "relu"})
            elif idx > 0 and idx < len(self.layers)-1:
                self.architecture.append({"input_dim": self.layers[idx-1].neurons, 
                                        "output_dim": layer.neurons,
                                        "activation": "relu"})    
            else:
                self.architecture.append({"input_dim": self.layers[idx-1].neurons,
                                            "output_dim": self.layers[idx].neurons,
                                            "activation": "softmax"})
        return self

    def _init_weights(self, data):
        self._compile(data)
        np.random.seed(42)
        for i, _ in enumerate(self.architecture):
            self.parameters.append(
                {
                    "W": np.random.uniform(low=-1, high=1,
                                       size=(self.architecture[i]["output_dim"],
                                            self.architecture[i]["input_dim"])),
                "b": np.zeros((1, self.architecture[i]["output_dim"]))
                }
            )
        return self

    def _forwardprop(self, data):
        A_curr = data
        for i, parameter in enumerate(self.parameters):
            A_prev = A_curr
            A_curr, Z_curr = self.layers[i].forward(inputs=A_prev,
                                                    weights=parameter["W"],
                                                    bias=parameter["b"],
                                                    activation=self.architecture[i]["activation"])
            self.memory.append({"inputs": A_prev, "Z": Z_curr})
        return A_curr    
    
    def _backprop(self, predicted, actual):
        raise NotImplementedError
    
    def _update(self, learning_rate=0.01):
        raise NotImplementedError
    
    def _get_accuracy(self, predicted, actual):
        raise NotImplementedError
    
    def _calculate_loss(self, predicted, actual):
        raise NotImplementedError
    
    def train(self, x_train, y_train, epochs):
        raise NotImplementedError
    
    def print_parameters(self):
        for i, parameter in enumerate(self.parameters):
            print(f"W{i} = {parameter['W'].shape}  \tb{i} = {parameter['b'].shape}")
    

x, y = generate_training_set(1000)
model = Network()
model.add(DenseLayer(1))
model._init_weights(x)

print(model.architecture)
model.print_parameters()
