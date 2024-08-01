import numpy as np
import matplotlib.pyplot as plt

class Neuron_Networks:
    def __init__(self, momentum_rate, learning_rate, activation_function, layer, activation_layer, data_type, epoch):
        self.Fv = []
        self.momentum_rate = momentum_rate
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.layer = layer
        self.activation_layer = activation_layer
        self.data_type = data_type
        self.epoch = epoch
        self.weight, self.d_weight, self.bias, self.d_bias, self.local_gradient = self.initialize(layer)
        self.epoch_errors = []

    def initialize(self, layer):
        weights = []
        delta_weights = []
        biases = []
        delta_biases = []
        local_gradient = []

        for i in range(len(layer) - 1):
            weights.append(np.random.rand(layer[i + 1], layer[i]))
            delta_weights.append(np.zeros((layer[i + 1], layer[i])))
            biases.append(np.random.rand(layer[i + 1]))
            delta_biases.append(np.zeros(layer[i + 1]))
            local_gradient.append(np.zeros(layer[i + 1]))

        return weights, delta_weights, biases, delta_biases, local_gradient

    def activation_fn(self, V):
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-V))
        elif self.activation_function == "relu":
            return np.where(V > 0, V, 0.0)
        elif self.activation_function == "tanh":
            return np.tanh(V)
        elif self.activation_function == "linear":
            return V

    def delta_activation_fn(self, V):
        if self.activation_function == "sigmoid":
            return V * (1 - V)
        elif self.activation_function == "relu":
            return np.where(V > 0, 1.0, 0.0)
        elif self.activation_function == "tanh":
            return 1 - V ** 2
        elif self.activation_function == "linear":
            return np.ones_like(V)

    def forward(self, input):
        self.X = [input]
        self.Fv = [input]
        for i in range(len(self.layer) - 1):
            self.Fv.append(self.activation_fn(self.weight[i] @ self.Fv[i] + self.bias[i]))

    def back_propagation(self, design_output):
        for i in reversed(range(len(self.layer) - 1)):
            if i == len(self.layer) - 2: 
                error = design_output - self.Fv[i + 1]
                self.local_gradient[i] = error * self.delta_activation_fn(self.Fv[i + 1])
            else:
                self.local_gradient[i] = self.delta_activation_fn(self.Fv[i + 1]) * np.dot(self.weight[i + 1].T, self.local_gradient[i + 1])
     
            self.d_weight[i] = (self.momentum_rate * self.d_weight[i]) + (self.learning_rate * np.outer(self.local_gradient[i], self.Fv[i]))
            self.d_bias[i] = (self.momentum_rate * self.d_bias[i]) + (self.learning_rate * self.local_gradient[i])
            self.weight[i] += self.d_weight[i]
            self.bias[i] += self.d_bias[i]
        return np.sum(error ** 2) / 2

    def train(self, input, design_output, epoch):
        keep_error = []
        for N in range(epoch):
            er = 0
            for i in range(len(input)):
                self.forward(input[i])
                er += self.back_propagation(design_output[i])
            er /= len(input)
            keep_error.append(er)
            print(f"Epoch = {N + 1} | Root Mean Square = {er}")
        return keep_error

    def compute_confusion_matrix(self, y_true, y_pred):
        classes = np.unique(np.concatenate((y_true, y_pred)))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for true, pred in zip(y_true, y_pred):
            cm[true, pred] += 1
        return cm

    def plot_confusion_matrix(self, cm, classes):
        plt.figure(figsize=(10, 7))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], horizontalalignment="center",
                         color="white" if cm[i, j] > cm.max() / 2. else "black")
        plt.show()

    def test(self, input, design_output, type):
        actual_output = []
        for i in input:
            self.forward(i)
            actual_output.append(self.Fv[-1])

        if type == "classification":
            actual_output = [np.argmax(output) for output in actual_output]
            design_output = [np.argmax(output) for output in design_output]
            correct_predictions = np.sum(np.array(actual_output) == np.array(design_output))
            accuracy = correct_predictions / len(actual_output)
            cm = self.compute_confusion_matrix(design_output, actual_output)
            print(f"Total samples: {len(actual_output)}")
            print(f"Confusion Matrix: \n{cm}")
            print(f"Accuracy: {accuracy}")
            self.plot_confusion_matrix(cm, ['Class 0', 'Class 1'])
        else:
            actual_output = [element[0] for element in actual_output]
            er = np.mean([np.sum((ao - do) ** 2) / 2 for ao, do in zip(actual_output, design_output)])
            print(f"Error = {er}")
            plt.figure(figsize=(10, 8))
            plt.subplot(3, 1, 1)
            plt.plot(self.epoch_errors, label='Mean Square Error')
            plt.title('Mean Square Error')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.legend()

            plt.subplot(3, 1, 2)
            plt.plot(design_output, label='Design Output')
            plt.plot(actual_output, label='Actual Output', linestyle='--')
            plt.legend()
            plt.title('Actual vs Design Output')
            plt.xlabel('Sample Index')
            plt.ylabel('Output Value')

            plt.subplot(3, 1, 3) 
            plt.plot(actual_output, label='Actual Output')
            plt.plot(design_output, label='Design Output', linestyle='--')
            plt.legend()
            plt.title('Actual and Predicted Values with Sample Index')
            plt.xlabel('Sample Index')
            plt.ylabel('Output Value')

            plt.tight_layout() 
            plt.show() 

    def normalize(self, input):
        data_min = np.min(input, axis=0)
        data_max = np.max(input, axis=0)
        return (input - data_min) / (data_max - data_min)

    def load_data(self,data_type):
        if self.data_type == "regression":
            return self.load_regression_data()
        elif self.data_type == "classification":
            return self.load_classification_data()
        else:
            raise ValueError("Unsupported data type")

    def load_regression_data(self, filename='Flood_data_set.txt'):
        data = []
        with open(filename) as f:
            for line in f.readlines()[2:]:
                data.append([float(element) for element in line.split()])
        data = np.array(data)
        np.random.shuffle(data)
        data = self.normalize(data)
        inputs = data[:, :-1]
        design_outputs = data[:, -1]
        return inputs, design_outputs

    def load_classification_data(self, filename='cross.txt'):
        data = []
        with open(filename) as f:
            rows = f.readlines()
            for line in range(1, len(rows), 3):
                z = np.array([float(element) for element in rows[line].split()])
                zz = np.array([float(element) for element in rows[line + 1].split()])
                data.append(np.append(z, zz))
        data = np.array(data)
        np.random.shuffle(data)
        inputs = data[:, :-2]
        design_outputs = data[:, -2:]  
        return inputs, design_outputs
    
    def k_fold_cross_validation(self, X, Y, K):
        fold_size = len(X) // K
        errors = []
        confusion_matrices = []

        for i in range(K):
            start = i * fold_size
            end = start + fold_size
            X_train = np.concatenate((X[:start], X[end:]), axis=0)
            Y_train = np.concatenate((Y[:start], Y[end:]), axis=0)
            X_val = X[start:end]
            Y_val = Y[start:end]

            nn = Neuron_Networks(self.momentum_rate, self.learning_rate, self.activation_function, self.layer, self.activation_layer, self.data_type, self.epoch)

            self.epoch_errors = nn.train(X_train, Y_train, self.epoch)

            actual_output = []
            for j in range(len(X_val)):
                nn.forward(X_val[j])
                actual_output.append(nn.Fv[-1])

            if self.data_type == 'classification':
                actual_output = [0 if out[0] > out[1] else 1 for out in actual_output]
                Y_val = [0 if out[0] > out[1] else 1 for out in Y_val]
                conf_matrix = nn.compute_confusion_matrix(np.array(Y_val), np.array(actual_output))
                confusion_matrices.append(conf_matrix)

                validation_error = np.mean([np.sum((np.array(do) - np.array(ao)) ** 2) / 2 for do, ao in zip(Y_val, actual_output)])
            else:
                validation_error = np.mean([np.sum((ao - do) ** 2) / 2 for ao, do in zip(actual_output, Y_val)])
            
            errors.append(validation_error)
            print(f"Fold {i + 1} | Validation Error = {validation_error}")

        avg_error = np.mean(errors)
        print(f"Average Validation Error across {K} folds: {avg_error}")

        if self.data_type == 'classification':
            avg_conf_matrix = np.mean(confusion_matrices, axis=0)
            print("Average Confusion Matrix across folds:")
            print(avg_conf_matrix)
            return avg_error, avg_conf_matrix

        return avg_error

if __name__ == "__main__":
    momentum_rate = 0.9
    learning_rate = 0.001
    activation_function = 'sigmoid'
    layer = [8, 20, 1]  # Adjust based on your dataset
    activation_layer = ['sigmoid', 'sigmoid']
    data_type = 'regression'  # 'regression' or 'classification'
    epoch = 100
    K = 5 
    
    nn = Neuron_Networks(momentum_rate, learning_rate, activation_function, layer, activation_layer, data_type, epoch)
    
    X, Y = nn.load_data(data_type)
    
    if data_type == "classification":
        avg_error, aggregated_cm = nn.k_fold_cross_validation(X, Y, K)
        nn.plot_confusion_matrix(aggregated_cm, ['Class 0', 'Class 1'])
    
    nn.k_fold_cross_validation(X, Y, K)
    
    nn.test(X, Y, type=data_type)
