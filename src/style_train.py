import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import os 

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out
    
class ResNetWithCBAM(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ResNetWithCBAM, self).__init__()
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            self._attach_cbam(base_model.layer3),
            self._attach_cbam(base_model.layer4),
        )
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _attach_cbam(self, layer):
        for i in range(len(layer)):
            if hasattr(layer[i],'conv3'):
                out_channels = layer[i].conv3.out_channels
            elif hasattr(layer[i],'conv2'):
                out_channels = layer[i].conv2.out_channels
            else:
                raise ValueError("Unsupported layer type for CBAM attachment")
            layer[i].cbam = CBAM(layer[i].conv3.out_channels)
            orig_forward = layer[i].forward

            def new_forward(x, orig_forward=orig_forward, block=layer[i]):
                out = orig_forward(x)
                out = block.cbam(out)
                return out

            layer[i].forward = new_forward
        return layer

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

def create_resnet18(num_classes):
    base_model = models.resnet18(pretrained=True)
    return ResNetWithCBAM(base_model, num_classes)

def create_resnet50(num_classes):
    base_model = models.resnet50(pretrained=True)
    return ResNetWithCBAM(base_model, num_classes)

def efficientnet(num_classes):
    model = models.efficientnet_b0(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    return model

MODEL_REGISTRY = {
    "resnet18": create_resnet18,
    "resnet50": create_resnet50,
    "efficientnet": efficientnet,
}
def get_model(model_name, num_classes):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry.")
    return MODEL_REGISTRY[model_name](num_classes)
def evaluate_model(model, loader, criterion, device):
    model.eval()
    y_true=[]
    y_pred=[]
    total_loss = 0.0
    correct = 0
    total =0
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
    #print("\nconfusion matix is :\n")
    #print(confusion_matrix(y_true,y_pred))
    

    return total_loss / len(loader), correct / total

def train_model(train_dataset, val_dataset, test_dataset, model_name="resnet18",num_classes=5, num_epochs=10, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)


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
        #evaluate_model(model, train_loader,criterion = criterion, device = device)
        print("\n Evaluation on validatioan set:")
        evaluate_model(model, val_loader,criterion=criterion,device=device)
        


    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%")
    print("\n Evaluation on test set:")
    evaluate_model(model,test_loader,criterion=criterion,device=device)
    
    #save the model
    os.makedirs("saved_models",exist_ok=True)
    torch.save(model.state_dict(),"saved_models/{model_name}_calligraphy.pth")
    print("\nmodel saved successfully in saved_models/{model_name}_calligraphy.pth ")
    #error=get_misclassified_samples(model_name, train_loader,device)
    return model, train_losses, val_losses, test_loss, test_acc,#error

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
        model, train_losses, val_losses, test_loss, test_acc =train_model(
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
        #errors.append(
            #{
             #   "model_name": model_name,
              #  "error_samples": error,
               # "error_count": len(error)
            #}
            #)
        

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
        
    
 #if __name__== "__main__":
    
 
#   