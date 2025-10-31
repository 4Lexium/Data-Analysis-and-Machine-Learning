# importing libraries
import autograd.numpy as np 
from autograd import grad, elementwise_grad
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits, load_breast_cancer
%matplotlib inline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, label_binarize, PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay, mean_squared_error
import string

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin  # not actuall anymore
from sklearn.utils import resample
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

### Hardcoded Values
traintestrat = 0.2    # train-test split ratio
figpath = 'C:\DATA_ANALYSIS_TEST\Proj2_Figs'

# defining activation functions:

def ReLU(z):
    return np.where(z > 0, z, 0)

def ReLU_leaky(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def ReLU_der(z):
    return np.where(z > 0, 1, 0)

def ReLU_leaky_der(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

def linear(z):
    return z

def linear_der(z):
    return np.ones_like(z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]

def softmax_stable(z):
    '''
    avoiding overflow by subtracting the maximum logit per sample before exponentiating 
    '''
    z_stable = z - np.max(z, axis=1, keepdims=True)
    e_z = np.exp(z_stable)
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def sigmoid_der(z):
    return sigmoid(z)*(1-sigmoid(z))

def mse(predict, target, weights=None, regularization=None, lambdaVal=0.0):
    """
    Mean Squared Error loss with optional L1/L2 regularization.
    """
    base_loss = np.mean((predict - target) ** 2)
    if regularization == 'L2' and weights is not None:
        reg_term = lambdaVal * np.sum(weights ** 2)
        return base_loss + reg_term
    elif regularization == 'L1' and weights is not None:
        reg_term = lambdaVal * np.sum(np.abs(weights))
        return base_loss + reg_term
    return base_loss

def mse_der(predict, target):
    """
    Derivative of MSE with optional L1/L2 regularization.
    No regularization here (as its relevant for dCdw only, but there we make it an add-on in Backpropagation function)
    """
    return 2 * (predict - target) / predict.shape[0]

def accuracy(predictions, targets):
    """
    Compute classification accuracy (one-hot targets).
    """
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    return np.mean(pred_classes == true_classes)

def cross_entropy(predict, target, weights=None, regularization=None, lambdaVal=0.0):
    """
    Cross-entropy loss with optional L1/L2 regularization.
    """
    epsilon = 1e-12
    predict = np.clip(predict, epsilon, 1. - epsilon)
    base_loss = -np.sum(target * np.log(predict)) / predict.shape[0]
    if regularization == 'L2' and weights is not None:
        base_loss += lambdaVal * np.sum(weights ** 2)
    elif regularization == 'L1' and weights is not None:
        base_loss += lambdaVal * np.sum(np.abs(weights))
    return base_loss

def cross_entropy_der(predict, target):
    """
    Derivative for softmax + cross-entropy combination
    Regularization effect is inside the backpropagation function as a seperate add-on
    """
    return (predict - target) / predict.shape[0]

def BATCH_feed_forward(input, layers, activation_funcs):
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a@W+b                     # flipped a, W
        a = activation_func(z)
    return a

def BATCH_feed_forward_saver(input, layers, activation_funcs):
    '''
    modification: because backprop needs a save of all FeedForward calculations (z) to be used later for the sigmas 
    '''
    layer_inputs = []
    zs = []
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a@W + b    #flipped a, W
        a = activation_func(z)
        zs.append(z)
    return layer_inputs, zs, a

def BATCH_cost(layers, input, activation_funcs, target, cost_type='entropy', regularization=None, lambdaVal=0.0):
    predict = BATCH_feed_forward(input, layers, activation_funcs)
    if cost_type == 'entropy':
        return cross_entropy(predict, target, weights=np.concatenate([W.flatten() for W, _ in layers]), regularization=regularization,lambdaVal=lambdaVal)
    elif cost_type == 'mse':
        return mse(predict, target, weights=np.concatenate([W.flatten() for W, _ in layers]), regularization=regularization, lambdaVal=lambdaVal)
    
def BATCH_backpropagation(input, layers, activation_funcs, target, activation_ders, cost_der=cross_entropy_der, regularization=None, lambdaVal=0.0):
    """
    Batched backpropagation with L1/L2 regularization mod.
    """
    layer_inputs, zs, predict = BATCH_feed_forward_saver(input, layers, activation_funcs)
    layer_grads = [() for layer in layers]
    # We loop over the layers, from the last to the first
    dC_dz_next = None  # Store gradient from next (up-lvl) layer
    for i in reversed(range(len(layers))):
        W, b = layers[i]  #current layer
        layer_input, z, activation_der_func = layer_inputs[i], zs[i], activation_ders[i]
        activation_der_values = activation_der_func(z)
        if i == len(layers) - 1:
            # For last layer: dC_dz = cost_der*activation_der_values
            # dC_da = cost_der(predict, target)
            # dC_dz = dC_da*activation_der_values
            if cost_der == cross_entropy_der:
                dC_dz = cost_der(predict, target)  #this is a combo for cross entropy and softmax!!!
            elif cost_der == mse_der:
                dC_da = cost_der(predict, target)
                dC_dz = dC_da*activation_der_values
        else:
            # For hidden layers: (inheritance) @ (up-lvel W) * (new activation derivative)
            # change for batched input!
            # dC_dz = dC_dz_next@W_next.T  * activation_der_values
            W_next, b_next = layers[i + 1] #downstream layer
            dC_da = dC_dz_next@W_next.T
            dC_dz = dC_da * activation_der_values
        # Compute gradients for this layer
        dC_dW = layer_input.T@dC_dz # changed for batched input
        dC_db = np.sum(dC_dz, axis=0) # changed to sum over batches
        # Regularization Add-on (using current layer W!)
        if regularization == 'L2':
            dC_dW += 2*lambdaVal*W
        elif regularization == 'L1':
            dC_dW += lambdaVal*np.sign(W)
        # Store this lvl 
        dC_dz_next = dC_dz
        layer_grads[i] = (dC_dW, dC_db)
    return layer_grads

class yagdi_NN:
    '''
    Yet Another Gradient Descent Index for Neural Networks
    '''
    def __init__(self, ProblemType='classification', learning_rate=0.01, optimizer='vanillagd', regularization=None, lambdaVal=0.0, max_iter=1000, max_epoch=100, tol=1e-6, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_type='batch', batch_size=32):
        self.problem_type = ProblemType
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.max_epoch = max_epoch
        self.optimizer = optimizer
        self.regularization = regularization
        self.lambdaVal = lambdaVal
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_type = batch_type
        self.batch_size = batch_size

        # Initialize state
        self.velocity_W = None
        self.velocity_b = None
        self.G_W = None
        self.G_b = None
        self.G2_W = None
        self.G2_b = None
        self.G3_W = None
        self.G3_b = None
        self.G4_W = None
        self.G4_b = None
        self.t = 0
    
    def initialize_for_layers(self, layers):
        '''
        Initialize optimizer state for the given layers
        '''
        self.velocity_W = [np.zeros_like(W) for W, b in layers]
        self.velocity_b = [np.zeros_like(b) for W, b in layers]
        self.G_W = [np.zeros_like(W) for W, b in layers]
        self.G_b = [np.zeros_like(b) for W, b in layers]
        self.G2_W = [np.zeros_like(W) for W, b in layers]
        self.G2_b = [np.zeros_like(b) for W, b in layers]
        self.G3_W = [np.zeros_like(W) for W, b in layers]
        self.G3_b = [np.zeros_like(b) for W, b in layers]
        self.G4_W = [np.zeros_like(W) for W, b in layers]
        self.G4_b = [np.zeros_like(b) for W, b in layers]
        self.t = 0
    
    def _get_batch(self, X, y):
        '''
        Select 1 random batch based on batch_type
        '''
        n_samples = X.shape[0]

        if self.batch_type == 'batch':
            return X, y  # Full dataset
        elif self.batch_type == 'stochastic':
            # Single random sample
            idx = np.random.randint(n_samples)
            return X[idx:idx+1], y[idx:idx+1]
        elif self.batch_type == 'minibatch':
            # ONE random mini-batch
            batch_size = min(self.batch_size, n_samples)
            indices = np.random.choice(n_samples, batch_size, replace=False)
            return X[indices], y[indices]
    
    def fit(self, X, y, layers, activation_funcs, activation_ders):
        '''
        Main training method - uses backpropagation internally
        compared to proj1, only 1 batch is evaluated er epoch 
        '''
        # Initialize optimizer variables ONCE
        self.initialize_for_layers(layers)
        
        total_iterations = 0
        converged = False
        
        # EPOCH LOOP 
        for epoch in range(self.max_epoch):
            if total_iterations >= self.max_iter:
                break
            if converged:
                break
            
            # Get ONE random batch per epoch
            X_batch, y_batch = self._get_batch(X, y)
            
            # Compute gradients using backpropagation
            if self.problem_type=='classification':
                gradients = BATCH_backpropagation(X_batch, layers, activation_funcs, y_batch, activation_ders, cost_der=cross_entropy_der, regularization=self.regularization, lambdaVal=self.lambdaVal)
            elif self.problem_type=='regression':
                gradients = BATCH_backpropagation(X_batch, layers, activation_funcs, y_batch, activation_ders, cost_der=mse_der, regularization=self.regularization, lambdaVal=self.lambdaVal)
            else:
                raise ValueError(f"Unknown problem type: {self.type}")
            
            # Apply optimizer update
            layers = self._apply_optimizer_update(layers, gradients)
            total_iterations += 1

            grad_norm = self._compute_gradient_norm(gradients)
            if grad_norm < self.tol:
                converged = True
                break

        return layers

    def _compute_gradient_norm(self, gradients):
        '''
        Compute norm of all gradients for convergence check
        '''
        total_norm = 0
        for dW, db in gradients:
            total_norm += np.sum(dW**2) + np.sum(db**2)
        return np.sqrt(total_norm)
    
    def _apply_optimizer_update(self, layers, gradients):
        '''
        Layer GD updates
        Note: with regularization inside gradients!
        '''
        updated_layers = []
        for layer_idx, ((W, b), (dW, db)) in enumerate(zip(layers, gradients)):

            if self.optimizer == 'vanillagd':
                W_new = W - self.learning_rate * dW
                b_new = b - self.learning_rate * db
                
            elif self.optimizer == 'momentum':
                self.velocity_W[layer_idx] = self.momentum * self.velocity_W[layer_idx] - self.learning_rate * dW
                self.velocity_b[layer_idx] = self.momentum * self.velocity_b[layer_idx] - self.learning_rate * db
                
                W_new = W + self.velocity_W[layer_idx]
                b_new = b + self.velocity_b[layer_idx]
                
            elif self.optimizer == 'adagrad':
                self.G_W[layer_idx] += dW**2
                self.G_b[layer_idx] += db**2
                
                lr_W = self.learning_rate / (np.sqrt(self.G_W[layer_idx]) + self.epsilon)
                lr_b = self.learning_rate / (np.sqrt(self.G_b[layer_idx]) + self.epsilon)
                
                W_new = W - lr_W * dW
                b_new = b - lr_b * db
                
            elif self.optimizer == 'rmsprop':
                self.G2_W[layer_idx] = self.beta1 * self.G2_W[layer_idx] + (1 - self.beta1) * dW**2
                self.G2_b[layer_idx] = self.beta1 * self.G2_b[layer_idx] + (1 - self.beta1) * db**2
                
                lr_W = self.learning_rate / (np.sqrt(self.G2_W[layer_idx]) + self.epsilon)
                lr_b = self.learning_rate / (np.sqrt(self.G2_b[layer_idx]) + self.epsilon)
                
                W_new = W - lr_W * dW
                b_new = b - lr_b * db
                
            elif self.optimizer == 'adam':
                self.t += 1
                
                # First moment
                self.G3_W[layer_idx] = self.beta1 * self.G3_W[layer_idx] + (1 - self.beta1) * dW
                self.G3_b[layer_idx] = self.beta1 * self.G3_b[layer_idx] + (1 - self.beta1) * db
                
                # Second moment
                self.G4_W[layer_idx] = self.beta2 * self.G4_W[layer_idx] + (1 - self.beta2) * dW**2
                self.G4_b[layer_idx] = self.beta2 * self.G4_b[layer_idx] + (1 - self.beta2) * db**2
                
                # Bias correction
                m_W_hat = self.G3_W[layer_idx] / (1 - self.beta1**self.t)
                m_b_hat = self.G3_b[layer_idx] / (1 - self.beta1**self.t)
                v_W_hat = self.G4_W[layer_idx] / (1 - self.beta2**self.t)
                v_b_hat = self.G4_b[layer_idx] / (1 - self.beta2**self.t)
                
                # Update
                W_new = W - self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
                b_new = b - self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            
            else:
                # Fallback to vanilla GD
                W_new = W - self.learning_rate * dW
                b_new = b - self.learning_rate * db
            
            updated_layers.append((W_new, b_new))
        
        return updated_layers 
    
class YANNI:
    def __init__(self, ProblemType='classification', network_architecture=None, activation_funcs=None, optimizer='adam', batch_type='minibatch', regularization=None, learning_rate=0.01, batch_size=32, lambdaVal=0.0, max_iter=1000, max_epoch=100, tol=1e-6, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        '''
        Yet Another Neural Network Index
        - Network architecture setup
        - Forward propagation  
        - Training in yagdi_NN optimizer (handles backpropagation internally)

        Paramters: 
            problem type: regression or classification, forwards the info to yagdi_NN (where it will mean mse or cross entropy)
            network_architecture: Layer sizes [input_size, hidden1_size, hidden2_size, ..., output_size]
            activation_funcs: list of function names (derivatives are mapped inside, just make sure the function are pre-defined)
            learning_rate, batch_size, max_iter, max_epoch, tol (defaults) for yagdi_NN
            provide: optimizer('adam', 'vanillagd', 'rmsprop', 'momentum'), batch_type('minibatch', 'stochastic', 'batch') and regularization('L2', 'L1')
            regulariation option unavailable
        '''
        self.ProblemType = ProblemType.lower()
        self.network_architecture = network_architecture
        self.activation_funcs = activation_funcs
        self.activation_ders = self._get_activation_derivatives(activation_funcs)
        
        # Initialize layers
        self.layers = self._create_layers(network_architecture)
        
        # Initialize optimizer (yagdi_NN handles backpropagation internally)
        self.optimizer = yagdi_NN(
            ProblemType=self.ProblemType,
            regularization=regularization,
            learning_rate=learning_rate,
            lambdaVal=lambdaVal,
            optimizer=optimizer,
            max_iter=max_iter,
            max_epoch=max_epoch,
            tol=tol,
            batch_type=batch_type,
            batch_size=batch_size,
            momentum=momentum,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon
        )

    def _to_onehot(self, y, n_classes=None):
        """Convert class labels to one-hot encoding"""
        if n_classes is None:
            n_classes = len(np.unique(y))
        return np.eye(n_classes)[y]
    
    def _create_layers(self, architecture):
        """Create network layers"""
        layers = []
        for i in range(len(architecture) - 1):
            input_size = architecture[i]
            output_size = architecture[i + 1]
            W = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
            b = np.random.randn(output_size) * np.sqrt(2 / input_size)
            layers.append((W, b))
        return layers
    
    def _get_activation_derivatives(self, activation_funcs):
        """Map activation functions to their derivatives"""
        derivative_map = {
            sigmoid: sigmoid_der,
            ReLU: ReLU_der,
            ReLU_leaky: ReLU_leaky_der,
            softmax_stable: lambda x: 1,  
            softmax: lambda x: 1,  
            linear: linear_der
            # tanh: tanh_der
        }
        return [derivative_map[func] for func in activation_funcs]
    
    def forward_propagation(self, X):
        """Forward pass through the network"""
        return BATCH_feed_forward(X, self.layers, self.activation_funcs)
    
    def compute_cost(self, X, y):
        """Compute cost on given data"""
        if self.ProblemType == 'regression':
            return BATCH_cost(layers=self.layers, input=X, activation_funcs=self.activation_funcs, target=y, cost_type='mse', regularization=self.optimizer.regularization, lambdaVal=self.optimizer.lambdaVal)
        elif self.ProblemType == 'classification':
            return BATCH_cost(layers=self.layers, input=X, activation_funcs=self.activation_funcs, target=y, cost_type='entropy', regularization=self.optimizer.regularization, lambdaVal=self.optimizer.lambdaVal)
    
    def compute_accuracy(self, X, y):
        """Compute accuracy on given data"""
        predictions = self.forward_propagation(X)
        predicted_classes = np.argmax(predictions, axis=1)

        # Handle both one-hot and class labels
        if y.ndim == 2 and y.shape[1] > 1:  # One-hot encoded
            true_classes = np.argmax(y, axis=1)
        else:  # Class labels (flatten to 1D array)
            true_classes = y.reshape(-1)
            
        return np.mean(predicted_classes == true_classes)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Complete training using yagdi_NN optimizer
        yagdi_NN handles backpropagation internally
        """
        if self.ProblemType == 'classification':
            # Convert to one-hot for training (works for both formats)
            if y_train.ndim == 1:  # Class labels
                y_train_onehot = self._to_onehot(y_train, n_classes=self.network_architecture[-1])
            else:  # Already one-hot
                y_train_onehot = y_train
        elif self.ProblemType == 'regression':
            y_train_onehot = y_train

        # Train using yagdi_NN (which calls BATCH_backpropagation internally)
        self.layers = self.optimizer.fit(
            X=X_train,
            y=y_train_onehot,
            layers=self.layers,
            activation_funcs=self.activation_funcs,
            activation_ders=self.activation_ders
        )
    
    def predict(self, X):
        """Make predictions on new data"""
        return self.forward_propagation(X)
    
    def predict_classes(self, X):
        """Return class predictions"""
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Return probability predictions (compatible with sklearn)"""
        return self.forward_propagation(X)
    
    def _ensure_class_labels(self, y):
        """Convert any format to class labels - UNIVERSAL"""
        if y.ndim == 2 and y.shape[1] > 1:  # One-hot
            return np.argmax(y, axis=1)
        else:  # Class labels
            return y.reshape(-1)
    
    def plot_confusion_matrix(self, X, y_true, normalize='true'):
        """Plot confusion matrix - UNIVERSAL"""
        y_pred = self.predict_classes(X)
        y_true_classes = self._ensure_class_labels(y_true)
        
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true_classes, y_pred,
            normalize=normalize,
            cmap=plt.cm.Blues,
            values_format='.2f'
        )
        plt.title('Normalized Confusion Matrix')
        plt.savefig(rf"{figpath}\{self.ProblemType}_ConfusionMatrix.png")
        plt.show()
    
    def plot_roc_curves(self, X, y_true):
        """Plot ROC curves - UNIVERSAL"""
        y_score = self.predict_proba(X)
        y_true_classes = self._ensure_class_labels(y_true)
        classes = np.unique(y_true_classes)
        n_classes = len(classes)
        
        # Binarize the labels for multiclass
        y_true_bin = label_binarize(y_true_classes, classes=classes)
        
        fig, ax = plt.subplots(figsize=(8, 6))

        if n_classes == 2:
            # For binary: y_true_bin has shape (n_samples,1), add the negative class
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        for class_idx in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_score[:, class_idx])
            roc_auc = auc(fpr, tpr)
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=f'Class {classes[class_idx]} (AUC = {roc_auc:.2f})')
            display.plot(ax=ax)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        ax.legend()
        plt.title('ROC Curves')
        plt.grid(True)
        plt.savefig(rf"{figpath}\{self.ProblemType}_ROCcurve.png")
        plt.show()
    
    def plot_cumulative_gain(self, X, y_true):
        """Plot cumulative gain curves - UNIVERSAL"""
        y_score = self.predict_proba(X)
        y_true_classes = self._ensure_class_labels(y_true)
        classes = np.unique(y_true_classes)
        n_classes = len(classes)
        
        plt.figure(figsize=(8, 6))
        for class_idx in range(n_classes):
            y_true_binary = (y_true_classes == classes[class_idx]).astype(int)
            sort_idx = np.argsort(y_score[:, class_idx])[::-1]
            y_sorted = y_true_binary[sort_idx]
            
            # Handle case where there are no positive samples for this class
            if np.sum(y_sorted) > 0:
                cumulative_gains = np.cumsum(y_sorted) / np.sum(y_sorted)
            else:
                cumulative_gains = np.zeros_like(y_sorted, dtype=float)
                
            percentage_population = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
            plt.plot(percentage_population, cumulative_gains, label=f'Class {classes[class_idx]}', lw=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', lw=2)
        plt.xlabel('Percentage of Sample')
        plt.ylabel('Gain')
        plt.title('Cumulative Gain Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(rf"{figpath}\{self.ProblemType}_CumGain.png")
        plt.show()
    
    def evaluate_model(self, X_test, y_test):
        """Run all evaluations at once - UNIVERSAL"""
        print("=== Model Evaluation ===")
        
        if self.ProblemType == 'classification':
            # Accuracy
            test_acc = self.compute_accuracy(X_test, y_test)
            print(f"Test Accuracy: {test_acc:.4f}")
            
            # Plots
            self.plot_confusion_matrix(X_test, y_test)
            self.plot_roc_curves(X_test, y_test)
            self.plot_cumulative_gain(X_test, y_test)
        
        elif self.ProblemType == 'regression':
            # MSE cost
            test_cost = self.compute_cost(X_test, y_test)
            print(f"Test Cost (MSE): {test_cost:.4f}")
            
            # Plot prediction vs true data
            plt.figure(figsize=(6, 6))
            y_pred = self.predict(X_test)
            plt.scatter(y_test, y_pred, alpha=0.7)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel("True Values")
            plt.ylabel("Predictions")
            plt.title("Regression Predictions vs True Values")
            plt.grid(True)
            plt.show()

def runge_function(N, noise=True, stdev=0.5):
    '''
    define runge function in the given interval with stochastic noise following N(0,sigma)
    sigma is a hardcoded value, 0.3 is recomended for the followign test cases.
    returns x and y vectors of size N
    '''
    x = np.linspace(-1, 1, N)
    y = 1/(1+25*x**2) 
    if noise:
        y += np.random.normal(0, stdev, size=N) 
    return x.reshape(-1,1), y.reshape(-1,1) # ensure column vectors

def pre_processing(x, y, test_ratio=0.2):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=42)
    
    # Save unscaled versions for plotting
    x_train, x_test = X_train.copy(), X_test.copy()

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Remove mean offset from y
    y_offset = np.mean(y_train, axis=0)
    y_train_scaled = y_train - y_offset
    
    # Maximal eigenvalue of Hessian (for learning rate heuristics)
    H = (2.0/X_train_scaled.shape[0]) * X_train_scaled.T @ X_train_scaled
    max_eig = np.max(np.linalg.eigvals(H))
    
    return X_train_scaled, X_test_scaled, x_train, x_test, y_train_scaled, y_offset, y_test, max_eig

def runge_function_2d(N, noise=True, stdev=0.5):
    '''
    Define 2D Runge function for proper 3D surface plotting
    '''
    grid_size = int(np.sqrt(N))
    x1 = np.linspace(-1, 1, grid_size)
    x2 = np.linspace(-1, 1, grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    # 2D Runge function: 1/(1 + 25*(x1^2 + x2^2))
    y = 1/(1 + 25*(X1**2 + X2**2))
    if noise:
        y += np.random.normal(0, stdev, size=y.shape)
    return np.column_stack([X1.ravel(), X2.ravel()]), y.ravel()