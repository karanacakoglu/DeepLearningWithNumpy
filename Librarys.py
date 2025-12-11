import numpy as np

#Temel Olusturma
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        pass
    def backward(self, output_gradient):
        pass

#Network Olusturma
class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()

        #He initialization uygulanacak varyans = 2  / input_size
        self.weights = np.random.randn(input_size,output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))

        # Gradyanları saklayacağımız yerler (Optimizer kullanacak)
        self.dweights = None
        self.dbias = None
    def forward(self, input_data):
        self.input = input_data
        #y = W * x
        self.output = np.dot(self.input, self.weights) + self.bias

        return self.output
    def backward(self, output_gradient):
        # 1. Ağırlıkların Gradyanı: dW = X^T . dY
        self.dweights = np.dot(self.input.T, output_gradient)

        # 2. Bias Gradyanı: db = Sum(dY)
        self.dbias = np.sum(output_gradient, axis=0, keepdims = True)

        # 3. Geriye gidecek hata: dX = dY . W^T
        input_gradient = np.dot(output_gradient, self.weights.T)
        return input_gradient

#3. Aktivasyon Fonksiyonları Olusturma

class ReLu(Layer):
    def forward(self, input_data):
        self.input = input_data
        # Negatifleri sıfırla
        return np.maximum(0, input_data)
    def backward(self, output_gradient):
        # Türev: Giriş > 0 ise 1, değilse 0
        # Gelen hata ile yerel türevi çarpıyoruz
        return output_gradient * (self.input > 0)

class SoftMax(Layer):
    def forward(self, input_data):
        # Stabilite için (sayısal patlamayı önlemek adına max değeri çıkarıyoruz)
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    def backward(self, output_gradient):
        # Not: Softmax türevini burada basitleştirilmiş haliyle geçiyoruz.
        # Genelde CrossEntropy ile birleşik hesaplanır.
        # Bu kısım tek başına kullanıldığında karmaşıklaşır,
        # o yüzden asıl türevi Loss fonksiyonunda "Softmax+Loss" türevi olarak halledeceğiz.
        return output_gradient

#4. Loss Function: Categorical Cross-Entropy

class CategoricalCrossEntropy:
    def forward(self, y_pred, y_true):
        _pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # Her satır için doğru sınıfın logaritmasını alıp topluyoruz
        correct_confidences = np.sum(y_pred * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

    def backward(self, y_pred, y_true):
        # Softmax + CrossEntropy ikilisinin türevi çok basittir: (Tahmin - Gerçek)
        # Bu formül zincirin ilk halkasıdır.
        return (y_pred - y_true) / len(y_pred)

#5. Optimizer (SGD + L1/L2 Regularization)

class SGD:
    def __init__(self, learning_rate=0.1, reg_lambda=0.01):
        self.lr = learning_rate
        self.reg_lambda = reg_lambda

    def update(self, layer):
        if hasattr(layer, 'weights'):
            # L2 Regularization (Weight Decay)
            layer.dweights += 2 * self.reg_lambda * layer.weights

            # Güncelleme
            layer.weights -= self.lr * layer.dweights
            layer.bias -= self.lr * layer.dbias