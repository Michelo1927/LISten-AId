import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y']


# Definisci la tua CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 22)  # 22 classi corrispondenti alle lettere dell'alfabeto tranne G, S, Z, J

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo in uso:", device)

    train_dataset = ImageFolder(root=r'train_test\train_data', transform=transform)
    test_dataset = ImageFolder(root=r'train_test\test_data', transform=transform)

 
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = CNN().to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    epochs = 13
    train_losses = []


    for epoch in range(epochs):  
        print('epoca '+ str(epoch) + ' iniziata')
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 100))
        
        average_loss = running_loss / len(train_loader)
        train_losses.append(average_loss)
        
            
        print('epoca '+ str(epoch) + ' terminata')
        
        print()

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []


    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())


    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    # Percorso di salvataggio del modello
    nome_modello = '/' + str(epochs)+'epochs.pth' 
    model_path = r'modello\v8'+nome_modello

    # Salva il modello
    torch.save(model.state_dict(), model_path)

    # Calcola la confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Funzione per plottare la confusion matrix con le lettere dell'alfabeto
    def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Plotta la confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cm, classes=class_names, normalize=False, title='Confusion matrix')
    plt.show()

    # Plotta la curva di apprendimento
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-', color='b')
    plt.title('Curva di Apprendimento')
    plt.xlabel('Epoca')
    plt.ylabel('Perdita')
    plt.grid()
    plt.show()