# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:PRAKASH C

### Register Number:212223240122

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    #Include your code here
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```

### Dataset Information

<img width="275" height="466" alt="image" src="https://github.com/user-attachments/assets/4101da63-3db9-4ecc-b812-dac932f3aee7" />



### OUTPUT

### Training Loss Vs Iteration Plot
<img width="332" height="185" alt="image" src="https://github.com/user-attachments/assets/2f6e8033-d3af-4719-93e2-742c9f33bb33" />

<img width="225" height="33" alt="image" src="https://github.com/user-attachments/assets/83193b6f-de2e-4106-b525-765b11ddd15e" />

<img width="680" height="467" alt="image" src="https://github.com/user-attachments/assets/ba01f069-01e6-482a-adfd-5c6e94e73219" />

### New Sample Data Prediction

<img width="344" height="27" alt="Screenshot 2026-02-12 212848" src="https://github.com/user-attachments/assets/91848685-800e-4c1f-aa47-ec5388a742cc" />



## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
