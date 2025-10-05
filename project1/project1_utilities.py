# list of standard libraries 
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin  # so our yagdi can function with pipeline
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

### Hardcoded Values has 
# If you wish to use to include utilities.py instead of .ipynb notebook: these values must replace the corresponding default values in function parameters. 
traintestrat = 0.2    # train-test split ratio
interceptFit = False  #exclude intercept in polynomial fit
figpath = '\Projects\project1_figures'

def polynomial_features(x, p, intercept=False):
    '''
    Inputs: data and model complexity (max polynomial degree)
    intercept = False: excluding interecept in training
    '''
    x = np.array(x) # so x is np.array and not list
    startat = 0 if intercept else 1
    return np.vstack([x**i for i in range(startat, p+1)]).T

def OLS_params(X, y):
    '''
    Use the standard OLS optimization method to find optimal coefficients
    numpy.psuedo inverse: SVD decomposition in case det(H)=0
    returns vector of size P
    '''
    # return np.linalg.inv(X.T @ X) @ X.T @ y
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def MSE(y_dat, y_model):
    '''
    returns Mean Squared Error
    '''
    n = np.size(y_model)
    return np.sum((y_dat-y_model)**2)/n

def Rsquared(y_dat, y_model):
    '''
    returns R2 score or MSE/variance(y_data)
    '''
    y_mean = np.mean(y_dat)
    return 1 - np.sum((y_dat-y_model)**2) / np.sum((y_dat-y_mean)**2)

def Ridge_params(X, y, l=None):
    '''
    OLS with regularization parameter l (default:0)
    numpy.psuedo inverse: SVD decomposition in case det(H)=0
    returns vector of size P
    '''
    # Assumes X is scaled and has no intercept column
    if not l:
        l = np.median(X.T @ X )/X.shape[1]
    # return np.linalg.inv(X.T @ X + l*np.eye(X.shape[1])) @ X.T @ y
    return np.linalg.pinv(X.T @ X + l*np.eye(X.shape[1])) @ X.T @ y

def runge_function(N, noise=True):
    '''
    define runge function in the given interval with stochastic noise following N(0,sigma)
    sigma is a hardcoded value, 0.3 is recomended for the followign test cases.
    returns x and y vectors of size N
    '''
    x = np.linspace(-1, 1, N)
    y = 1/(1+25*x**2) 
    if noise:
        y += np.random.normal(0, 0.3, size=N) 
    return x, y 

def pre_processing(x, y, N, P):
    '''
    standardize feature columns to have zero mean and unit variance
    removes the data offset before optimization (for data scale independancy)
    this ensures that each feature has equal weighting in the analysis
    performs train test split with 1:4 rato 
    returns: scaled X_train/test, (x_test and x_train for plotting), y_scaled(train)+y_offset, y_test and the maximal eigenvalue of the Hessian
    '''
    # Feature Scaling
    FM = polynomial_features(x, P, intercept=interceptFit)
    FM_mean = FM.mean(axis=0)
    FM_std = FM.std(axis=0)
    FM_std[FM_std == 0] = 1  # safeguard to avoid division by zero for constant features
    FM_sc = (FM - FM_mean) / FM_std
    # Get Hessian Eigenvalues 
    H = (2.0/N) *FM_sc.T @ FM_sc
    EigValues = np.linalg.eig(H)[0]
    max_eig = np.max(EigValues)
    
    # Sclae Data
    X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(FM_sc, y, x, test_size=traintestrat)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_offset = np.mean(y_train)
    y_sc = y_train - y_offset
    return X_train_scaled, X_test_scaled, x_train, x_test, y_sc, y_offset, y_test, max_eig   # substract offset when training, add it again for both test and train evaluation
    
def ols_analysis(X_train_scaled, X_test_scaled, y_sc, y_offset, y_test): 
    '''
    inputs: X_train/test, y_scaled (test) + y_offset, and y_test
    Performs the OLS analysis for part A. 
    Finds and returns the optimal coefficients with MSE and R2 
    '''                
    theta_OLS = OLS_params(X_train_scaled, y_sc)
    y_train_pred = X_train_scaled @ theta_OLS + y_offset
    y_test_pred = X_test_scaled @ theta_OLS + y_offset 
    return theta_OLS, MSE(y_sc+y_offset, y_train_pred), MSE(y_test, y_test_pred), Rsquared(y_sc+y_offset, y_train_pred), Rsquared(y_test, y_test_pred)
    
def ridge_analysis(X_train_scaled, X_test_scaled, y_sc, y_offset, y_test, l_val):
    '''
    inputs: X_train/test, y_scaled (test) + y_offset, and y_test
    Performs the Ridge analysis with regulariszation parameter l, for part A. 
    Finds and returns the optimal coefficients with MSE and R2 
    ''' 
    theta_ridge = Ridge_params(X_train_scaled, y_sc, l=l_val)
    y_train_pred = X_train_scaled @ theta_ridge + y_offset
    y_test_pred = X_test_scaled @ theta_ridge + y_offset 
    return theta_ridge, MSE(y_sc+y_offset, y_train_pred), MSE(y_test, y_test_pred), Rsquared(y_sc+y_offset, y_train_pred), Rsquared(y_test, y_test_pred)

def get_grad_OLS(theta, FM, y, n):
    '''
    gradient of the OLS expression
    '''
    return (2.0/n) * FM.T @ (FM @ theta - y)

def get_grad_Ridge(theta, FM, y, lam, n):
    '''
    gradient of the Ridge expression with L2 regularization
    '''
    return get_grad_OLS(theta, FM, y, n) + 2*lam*theta    #/n

def simple_gradient_descend(X_train_scaled, X_test_scaled, y_sc, y_offset, y_test, max_eig, eta_vec, mode, l_val):
    '''
    inputs: max eig value of the Hessian (X^T@X) and a scale vector to find learning rates < 2/max_eig
            mode: OLS or Ridge
            l_val: regularization parameter
    Interatively calculates the optimal parameters using the Gradient Descend scheme
    Exit condition: max iteration or convergence (linear norm of the gradient < tolerance)
    returns a 2D list: [eta scale][iteration] 
    '''
    eta_vec_scaled = eta_vec / max_eig
    max_iter = 1000
    threshold = 1e-6
    N, degree = X_train_scaled.shape
    theta0 = 2.5*np.ones(degree) 
    log_Ridge = []

    # plase dont mind that varianbles are all called Ridge (content depends on 'mode')
    for eta in eta_vec_scaled:
        theta_Ridge = theta0.copy() 
        log_eta_Ridge = [theta_Ridge.copy()]
                  
        for iter in range(max_iter):
            if mode=='ols':
                grad_Ridge = get_grad_OLS(theta_Ridge, X_train_scaled, y_sc, N)
            elif mode=='ridge':
                grad_Ridge = get_grad_Ridge(theta_Ridge, X_train_scaled, y_sc, l_val, N)
            if np.linalg.norm(grad_Ridge) < threshold:
                break
            theta_Ridge -= grad_Ridge*eta
            log_eta_Ridge.append(theta_Ridge.copy())
            # print(theta_Ridge)
        log_Ridge.append(np.array(log_eta_Ridge))
    return log_Ridge

class yagdi(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=0.01, max_iter=1e6, tol=1e-6, optimizer='vanillagd', mode='ols', lambda_val=0.0, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_type='batch', batch_size=64, max_epoch=50, track_coef_history=True):
        """
        Yet Another Gradient Descent Index (YAGDI)
        
        Parameters:
        -----------
        optimizer: 'vanillagd', 'momentum', 'adagrad', 'rmsprop', 'adam'
        mode: 'ols', 'ridge', 'lasso'
        batch_type: 'batch', 'stochastic', 'minibatch', 'customsample' (imporance sampling)
        hyper parameters: regularization lambda, g^1 moment: momentum, beta1, g^2 moment: beta2, epsilon: singularity guard, batchsize=2^n
        exit conditions: max_iter, max_epoch, tol
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.optimizer = optimizer
        self.mode = mode
        self.lambda_val = lambda_val
        self.momentum = momentum  # For momentum optimizer
        self.beta1 = beta1        # For Adam/RMSprop first moment
        self.beta2 = beta2        # For Adam second moment
        self.epsilon = epsilon
        self.batch_type = batch_type
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.track_coef_history = track_coef_history
        self.coef_history_ = [] 
        self.N_full = 0
    
    #=========================================================================
    #=========================================================================
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.N_full = n_samples
        self.coef_ = np.ones(n_features)*2.5     # hardcoded 2 here <--------!!!
        self.intercept_ = 0.0
        self.loss_history_ = []
        if self.track_coef_history:
            self.coef_history_ = [self.coef_.copy()]
            
        # Initialize optimizer variables
        self._initialize_optimizer_variables(n_features)
        
        # Setup Epoch
        if self.batch_type == 'batch':
            for i in range(self.max_iter):
                # Its own sub process independant of epoch!
                X_batch, y_batch = self._get_batch(X, y, i)
                gradient = self._compute_gradient(X_batch, y_batch)
                current_loss = self._compute_loss(X, y)
                self.loss_history_.append(current_loss)
                if np.linalg.norm(gradient) < self.tol:
                    break
                self._apply_optimizer_update(gradient, i)
                if self.track_coef_history:
                    self.coef_history_.append(self.coef_.copy())
            return self 
        
        elif self.batch_type == 'stochastic':
            batches_per_epoch = n_samples
            batch_size = 1
        elif self.batch_type == 'minibatch':
            batches_per_epoch = n_samples // self.batch_size
            if n_samples % self.batch_size != 0:
                batches_per_epoch += 1
            batch_size=self.batch_size
        elif self.batch_type == 'customsample':
            batches_per_epoch = n_samples
            batch_size = 1
       
        total_iterations = 0
        converged = False
        
        for epoch in range(self.max_epoch):
            
            if total_iterations >= self.max_iter:
                break
            if converged:
                break
            
            # Shuffle data at the start of each epoch (for stochastic and minibatch) 
            if self.batch_type != 'batch':
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
                
            for batch_idx in range(batches_per_epoch):
                # Calculate start/end index for this batch
                start_idx = batch_idx*batch_size
                end_idx = min(start_idx + batch_size, n_samples) # Handle last batch
                
                # Get the batch from the shuffled data
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Compute gradient (BATCH gradient with/without regularization and with/without updating learning rate)
                gradient = self._compute_gradient(X_batch, y_batch)
                self._apply_optimizer_update(gradient, total_iterations) 
                total_iterations += 1
                
                if self.track_coef_history:
                    self.coef_history_.append(self.coef_.copy())
                
                if total_iterations >= self.max_iter:
                    break
                
            full_gradient = self._compute_gradient(X, y)
            current_loss = self._compute_loss(X, y)
            self.loss_history_.append(current_loss)
            if np.linalg.norm(full_gradient) < self.tol:
                converged = True
                                
        return self
    
    #=========================================================================
    #=========================================================================

    def _initialize_optimizer_variables(self, n_features):
        """Initialize variables for different optimizers"""
        if self.optimizer == 'momentum':
            self.previous_update = np.zeros(n_features) # setup memtory
        elif self.optimizer == 'adagrad':
            self.G = np.zeros(n_features)  # Cumulative sum of square garadiets
        elif self.optimizer == 'rmsprop':
            self.G2 = np.zeros(n_features) # Running average of suqare gradients
        elif self.optimizer == 'adam':
            self.G3 = np.zeros(n_features) # First moment vector (m)
            self.G4 = np.zeros(n_features) # second moment vector (v)
            self.t = 0 # time step
    
    def _get_batch(self, X, y, iteration):
        """Get appropriate batch based on batch_type"""
        n_samples = X.shape[0]
        
        if self.batch_type == 'batch':
            # Full dataset
            return X, y
            
        elif self.batch_type == 'stochastic':
            # Single random sample
            idx = np.random.randint(n_samples)
            return X[idx:idx+1], y[idx:idx+1]
        
        elif self.batch_type == 'customsample':
            # Importance Sampling defined by custom g(x) = tails of normal distribution
            if not hasattr(self, '_custom_probs'):
                # calculate and cache custom probabilities 
                x_values = X[:, 0]
                
                # # Option A. U-shaped distribution
                # x_normalized = 2 * (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values)) - 1
                # self._custom_probs = 1 + x_normalized**2 
                # self._custom_probs = self._custom_probs / np.sum(self._custom_probs) #normalize
                
                # Option B Inverted Normal distribution
                # Fit normal distribution to data
                mu, sigma = np.mean(x_values), np.std(x_values)    
                # normal_probs = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x_values - mu)/sigma)**2)
                # # Invert to get higher probability for tails
                # self._custom_probs = 1 / (normal_probs + 1e-10)  # Avoid division by zero
                # self._custom_probs = self._custom_probs / np.sum(self._custom_probs)
            
                # Sample from middle 60% of distribution, avoid extreme edges
                in_middle = np.abs(x_values - mu) < 0.6 * sigma
                middle_indices = np.where(in_middle)[0]
                middle_probs = np.ones(len(middle_indices)) / len(middle_indices)
            idx = np.random.choice(middle_indices, p=middle_probs)
            # idx = np.random.choice(n_samples, p=self._custom_probs)
            return X[idx:idx+1], y[idx:idx+1]
            
        elif self.batch_type == 'minibatch':
            # Random mini-batch
            batch_size = min(self.batch_size, n_samples)
            indices = np.random.choice(n_samples, batch_size, replace=False)
            return X[indices], y[indices]
    
    #======================================================================================================
    # Gradients
    #======================================================================================================
    
    def _compute_gradient(self, X, y):
        """Compute gradient with regularization"""
        # n_samples = self.N_full #full dataset
        n_samples = X.shape[0] #batch size)
        residuals = X @ self.coef_ - y
        
        # OLS gradient
        ols_gradient = (2 / n_samples) * X.T @ residuals
        
        # Add regularization
        if self.mode == 'ridge' and self.lambda_val > 0:
            ridge_gradient = (2.0)*self.lambda_val*self.coef_   #(2.0/n_samples)   
            return ols_gradient + ridge_gradient
            
        elif self.mode == 'lasso' and self.lambda_val > 0:
            # L1 gradient: lambda * sign(theta)
            lasso_gradient = (1.0)*self.lambda_val*np.sign(self.coef_)  #(1.0/n_samples)
            # Handle theta=0 
            lasso_gradient[self.coef_ == 0] = 0       # =self.lambda_val * np.random.choice([-1, 1], size=np.sum(self.coef_ == 0))
            return ols_gradient + lasso_gradient
        
        return ols_gradient
    
    def _compute_loss(self, X, y):
        """Compute current loss for monitoring"""
        n_samples = X.shape[0]
        residuals = X @ self.coef_ - y
        mse_loss = np.mean(residuals**2)
        
        if self.mode == 'ridge' and self.lambda_val > 0:
            return mse_loss + self.lambda_val * np.sum(self.coef_**2)   #L2 of theta times lambda (regularization)
        elif self.mode == 'lasso' and self.lambda_val > 0:
            return mse_loss + self.lambda_val * np.sum(np.abs(self.coef_)) 
        
        return mse_loss
    
    #======================================================================================================
    # Learning Rate Modifications
    #======================================================================================================
    
    def _apply_optimizer_update(self, gradient, iteration):
        """Apply optimizer-specific update rule"""
        if self.optimizer == 'vanillagd':
            # Vanilla gradient descent
            self.coef_ -= self.learning_rate * gradient
            
        elif self.optimizer == 'momentum':
            # current update = -learning rate*gradient + momentum parameter*previous update
            current_update = self.learning_rate*gradient + self.momentum*self.previous_update
            self.coef_ -= current_update
            self.previous_update = current_update
            
        elif self.optimizer == 'adagrad':
            # update G with new gradient squared (cumulative gradient)
            # udate theta with - learning rate * new gradient / sqrt(new G val + epsilon const)
            self.G += gradient**2
            adjusted_lr = self.learning_rate / (np.sqrt(self.G) + self.epsilon)
            self.coef_ -= adjusted_lr*gradient
            
        elif self.optimizer == 'rmsprop':
            # update G2 not simply += grad**2but a running averadge: beta*G2+(1-beta)*grad**2
            self.G2 = self.beta1*self.G2 + (1 - self.beta1)*gradient**2
            adjusted_lr2 = self.learning_rate / (np.sqrt(self.G2) + self.epsilon)
            self.coef_ -= adjusted_lr2*gradient 
            
        elif self.optimizer == 'adam':
            # update G3 and G4 like in rmsprop using beta1 and beta2, but G4 has gradient**2
            self.t += 1
            self.G3 = self.beta1 * self.G3 + (1 - self.beta1) * gradient
            self.G4 = self.beta2 * self.G4 + (1 - self.beta2) * gradient**2
            
            # Bias correction (use t*beta as beta_t)
            G3_hat = self.G3 / (1 - self.beta1**self.t)
            G4_hat = self.G4 / (1 - self.beta2**self.t)
            
            self.coef_ -= self.learning_rate * G3_hat / (np.sqrt(G4_hat) + self.epsilon)
    
    def predict(self, X, y_offset):
        return X @ self.coef_ + y_offset

def get_bias(y_true, y_pred):
    '''
    inputs: y_true: vector of size N
            y_pred: 2D array: (n_bootstraps, N_samples)
    returns: float value
    '''
    f_bar = np.mean(y_pred, axis=0)
    return np.mean((f_bar - y_true)**2)

def get_var(y_pred):
    '''
    inputs: y_pred: 2D array: (n_bootstraps, N_samples)
    returns: float value
    '''
    f_bar = np.mean(y_pred, axis=0)
    return np.mean((f_bar - y_pred)**2)

def get_mse(y_true, y_pred):
    '''
    inputs: y_true: vector of size N
            y_pred: 2D array: (n_bootstraps, N_samples)
    returns: float value
    '''
    return np.mean((y_pred - y_true.reshape(1, -1))**2)