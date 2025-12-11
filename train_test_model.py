import numpy as np

from Librarys import SoftMax, Dense, ReLu, SGD, CategoricalCrossEntropy

print("Veri hazırlanıyor...")
np.random.seed(42)
X = np.zeros((300, 3))
y_true = np.zeros((300, 3))

# Class 0 verileri
X[:100] = np.random.randn(100, 3) - 2
y_true[:100, 0] = 1

# Class 1 verileri
X[100:200] = np.random.randn(100, 3)
y_true[100:200, 1] = 1

# Class 2 verileri
X[200:] = np.random.randn(100, 3) + 2
y_true[200:, 2] = 1

# Karıştıralım (Shuffle)
keys = np.array(range(300))
np.random.shuffle(keys)
X = X[keys]
y_true = y_true[keys]

# 2. Model Kurulumu
network = [
    Dense(3, 64),
    ReLu(),
    Dense(64, 3),
    SoftMax()
]

optimizer = SGD(learning_rate=0.1, reg_lambda=0.001)  # L2'yi biraz kıstım
loss_func = CategoricalCrossEntropy()

# 3. Eğitim Döngüsü
print(f"Eğitim Başlıyor (Veri Boyutu: {len(X)})")

for epoch in range(1001):
    # Forward
    output = X
    for layer in network:
        output = layer.forward(output)

    loss = loss_func.forward(output, y_true)

    # Backward
    grad = loss_func.backward(output, y_true)
    for layer in reversed(network):
        grad = layer.backward(grad)

    # Update
    for layer in network:
        optimizer.update(layer)

    if epoch % 100 == 0:
        predictions = np.argmax(output, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        acc = np.mean(predictions == true_labels)
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}")

# Sonucu Görelim
print("\n--- TEST SONUCU ---")
test_input = np.array([[-2, -2, -2], [0.1, 0.1, 0.1], [2, 2, 2]])
out = test_input
for layer in network:
    out = layer.forward(out)
