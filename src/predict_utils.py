import torch
from torchvison import models,transforms
from PIL import Image
import csv 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_class_mapping(file_path):
    mapping ={}
    with open(file_path,"r",encoding="uft-8") as f:
        for row in csv.reader(f):
            idx,label = int(row[0]),row[1]
            mapping[idx] = label 
            
    return mapping 

def load_model (model_path, num_classes =5):
    model = model.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_path,mapping):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
        

    ])
    image= Image.open(image_path).convert("RGB")
    image= transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs= model(image)
        _, pred = torch.max(outputs,1)
    return mapping[pred.item()]


