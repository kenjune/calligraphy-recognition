import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import os 
def create_resnet18(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
def create_resnet50(num_classes):
    model = models.resnet50(pretrained=True)
    model.fc=nn.Linear(model.fc.in_features,num_classes)
    return model 
def create_vgg16(num_classes):
    model = models.vgg16(pretrained = True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features,num_classes)
    return model
MODEL_REGISTRY ={
    "resnet18" : create_resnet18,
    "resnet50" : create_resnet50,
    "vgg16" : create_vgg16,

}
def get_model(model_name, num_classes):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f" Model {model_name} not found in registry.")
    return MODEL_REGISTRY[model_name](num_classes)
def evaluate_model(model, loader, criterion, device):
    model.eval()
    y_true=[]
    y_pred=[]
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    print("\nclassification report is:\n")
    print(classification_report(y_true,y_pred))
    print("\nconfusion matix is :\n")
    print(confusion_matrix(y_true,y_pred))
    

    return total_loss / len(loader), correct / total

def train_model(train_dataset, val_dataset, test_dataset, model_name="resnet18",num_classes=5, num_epochs=10, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss)

        print(f"Train Loss: {running_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")
        evaluate_model(model, train_loader,criterion = criterion, device = device)
        evaluate_model(model, val_loader,criterion=criterion,device=device)
        


    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%")
    print("\n Evaluation on test set:")
    evaluate_model(model,test_loader,criterion=criterion,device=device)
    
    #save the model
    os.makedirs("saved_models",exist_ok=True)
    torch.save(model.state_dict(),"saved_models/{model_name}_calligraphy.pth")
    print("\nmodel saved successfully in saved_models/{model_name}_calligraphy.pth ")
    error=get_misclassified_samples(model_name, train_loader,device)
    return model, train_losses, val_losses, test_loss, test_acc,error

def get_misclassified_samples(model,data_loader,device):
    model.eval()
    errors = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs,1)
            for i in range(len(labels)):
                if labels[i] != predicted[i]:
                    errors.append((inputs[i],labels[i].item(), predicted[i].item()))
    return errors
def test_multiple_models(train_dataset, val_dataset, test_dataset, model_names,num_classes=5, num_epochs=10, batch_size=32, lr=0.001 ):
    results = []
    errors = []
    for model_name in model_names:
        print(f"\n Training and evaluating model: {model_name}")
        model, train_losses, val_losses, test_loss, test_acc, error =train_model(
            train_dataset, val_dataset, test_dataset, model_name=model_name, 
            num_classes = num_classes,
            num_epochs = num_epochs,batch_size = batch_size,lr=lr

        )
        results.append(
            {
                "model_name": model_name,
                "test_loss": test_loss,
                "test_acc": test_acc,


            }
        )
        errors.append(
            {
                "model_name": model_name,
                "error_samples": error,
                "error_count": len(error)
            }
            )
        

    print("\n Comparison of models:")
    for result in results:
        print(f"Model:{result['model_name']} | Test lossess: {result['test_loss']:.4f}| Test accuracy:{result['test_acc']*100:.2f}%")
    

    return results

class MisclassifiedDataset(Dataset):
    def __init__(self,error_samples, transform = None):
        self.samples = error_samples
        self.transform = transform 
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
            return img, label 
        
    
 if __name__== "__main__":
    
 
#     