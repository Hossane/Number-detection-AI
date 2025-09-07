import torch
from torch import nn
from torch.nn import functional

# The target accuracy, we stop training when we reach this accuracy
EPOCH_BREAK_ACCURACY = .995

# The number of epochs to train for
TEST_BATCH_SIZE = 1000

# Create a class for the model
class CNN(nn.Module):
  # Initialize the function
  def __init__(self):
    #
    super(CNN, self).__init__()
    # Convolution layer
    # Which is a layer that applies a filter to the image
    self.convl = nn.Conv2d(1,32,3,1)
    self.conv2 = nn.Conv2d(32,64,3,1)

    # dropout layer to prevent overfitting
    # overfitting is when the model is too complex and learns the training data too well
    # this is bad because if it is given new data, it will not be able to generalize well
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)

    # fully connected layer
    # to condense the neurons into a smaller layer
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)

  # define the forward function
  # this is where we define the flow of the data through the model, use the defined layers and create a flow of data
  def forward(self, x):
    x = self.convl(x)
    x = functional.relu(x) # activation function
    x = self.conv2(x)
    x = functional.relu(x)
    x = functional.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = functional.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    return x
  

def train_model(model, device, data_loader, loss_func, optimizer, num_epochs):
  train_loss, train_acc = [], []
  for epoch in range(num_epochs):
    runningLoss = 0.0
    correct = 0
    total = 0
    for images, labels, in data_loader:
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(images)
      loss = loss_func(outputs, labels)
      loss.backward()
      optimizer.step()

      runningLoss += loss.item()
      _, predicted = outputs.max(1)
      correct += predicted.eq(labels).sum().item()
      total += labels.size(0)
    epoch_loss = runningLoss / len(data_loader)
    epoch_acc = correct / total

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    if epoch_acc >= EPOCH_BREAK_ACCURACY:
      print(f"Model has reached {EPOCH_BREAK_ACCURACY} accuracy, training has been stopped")
      break
  return train_loss, train_acc

def test_model(model, data_loader, device=None):
  
  if device is None:
    device = torch.device('cpu')

  model.eval()
  test_loss = 0
  correct = 0

  data_len = len(data_loader.dataset)

  with torch.no_grad():
    for data, target in data_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += functional.cross_entropy(output, target, reduction='sum').item()
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(data_loader.dataset)
  accuracy = correct / data_len
  print(f"Test Set: Average Loss: {test_loss:.4f}, Accuracy: {correct}/{data_len} ({100 * accuracy}%)")
  return accuracy