import torch
import torchvision.transforms as transforms
import cv2
import os
from PIL import Image
from torch import nn
from torchvision.datasets import ImageFolder

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


# Percorso del file del modello suggerito '13epochs.pth'
model_path = r"modello\v8\13epochs.pth"

# Definisci le trasformazioni per pre-elaborare i frame
transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Lista delle lettere dell'alfabeto corrispondenti alle classi del modello
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo in uso:", device)

# Carica il modello
model = CNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Modello caricato correttamente")




def preprocess_image(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = transform(img).unsqueeze(0).to(device)
    return img


# percorso = r"train_test\test_data"

# for imm in os.listdir(percorso):
#     stringa = percorso + "\\" + imm
#     print(imm)
#     img = cv2.imread(stringa)
#     img = preprocess_image(img)
#     with torch.no_grad():
#         outputs = model(img)
#         output = torch.softmax(outputs, dim=1)
#         preds = torch.argmax(output, 1)
#         predicted_label = preds.item()
#         # print(preds)
#         # print(predicted_label)
#         print(class_names[predicted_label])





# Cattura il video dalla telecamera del PC
cap = cv2.VideoCapture(0)  # Imposta 0 per la telecamera predefinita
while True:
    ret, frame = cap.read()
    if not ret:
        break


    # Calcola le dimensioni dell'area di interesse al centro dello schermo
    height, width, _ = frame.shape
    size = min(height, width) // 2  
    center_x, center_y = width // 2, height // 2  # Coordinate del centro dello schermo


    cv2.rectangle(frame, (center_x - size, center_y - size), (center_x + size, center_y + size), (0, 255, 0), 2)  # Rettangolo
    
    # Ritaglia l'area di interesse dal frame
    cropped_frame = frame[center_y - size:center_y + size, center_x - size:center_x + size]
    
    #APPLICO LA FUNZIONE preprocess_image
    input_tensor = preprocess_image(cropped_frame)


    with torch.no_grad():
        output = model(input_tensor)
        softmax = torch.softmax(output, dim=1)

        predicted = torch.argmax(softmax, dim=1)

        predicted_letter = class_names[predicted]  # Usa la mappa delle lettere effettive
    
    # Visualizza il risultato sulla finestra del video
    cv2.putText(frame, predicted_letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Sign Language Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Esci se premi 'q'
        break

# Rilascia la cattura e chiudi le finestre
cap.release()
cv2.destroyAllWindows()
