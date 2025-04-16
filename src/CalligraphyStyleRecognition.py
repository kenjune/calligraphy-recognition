import csv
import os
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split 
num_class = 5
num_epochs = 10
batch_size = 32
lr = 0.001
# 训练数据存放地址

def get_train_data_path(train, csv_file_path, model_path):
    global train_data_pth, csv_file, model_save_pth
    train_data_pth = train
    csv_file = csv_file_path
    model_save_pth = model_path
    
#train_data_pth = 'E:/Chinese Calligraphy Script Styles/train' 

# 全局下标映射的csv文件

#csv_file = 'book_style.csv'
# 模型存放文件
#model_save_pth = 'resnet18_book_style.pth'
 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
# 引入resnet18网络模型
def create_resnet18():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)
    return model
 
 
def train_and_save_mode():
    # 加载原始训练集
    full_dataset = datasets.ImageFolder(root=train_data_pth, transform=transform)

    # 划分 train 和 val 索引
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=full_dataset.targets, random_state=42)

    # 创建子集
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 建立标签索引并保存为 CSV
    class_to_idx = full_dataset.class_to_idx
    class_dict = {val: key for key, val in class_to_idx.items()}
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for idx, label in class_dict.items():
            writer.writerow([idx, label])

    model = create_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss_list.append(running_loss / len(train_loader))
        train_acc_list.append(100. * correct / total)

        # 验证集评估
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_loss_list.append(val_loss / len(val_loader))
        val_acc_list.append(100. * correct / total)

    # 保存模型
    torch.save(model.state_dict(), model_save_pth)

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list

 
 
# 加载全局下标映射
def load_class_mapping():
    class_to_idx = {}
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            idx = int(row[0])
            class_name = row[1]
            class_to_idx[idx] = class_name
    return class_to_idx
 
 
def load_model():
    if not os.path.isfile(model_save_pth):
        return False
    else:
        model = models.resnet18().to(device)
        model.fc = nn.Linear(model.fc.in_features, num_class)
        model.load_state_dict(torch.load(model_save_pth))
        model.eval()  # 设置模型为评估模式
        return model
 
 
def predict_image(image, model):
    # image = Image.fromarray(image_array)  # 打开图像
    image = transform(image).unsqueeze(0)  # 预处理并增加批次维度
    image = image.to(device)
    model = model.to(device)
    with torch.no_grad():  # 禁用梯度计算
        output = model(image)
        _, predicted = torch.max(output, 1)  # 获取预测类别
    return predicted.item()  # 返回类别索引
 
 
# 对外调用的字形识别方法
def book_style_predict(image_array):
    class_dict = load_class_mapping()
    trained_model = load_model()
    if not trained_model:
        return False
    else:
        img = Image.fromarray(image_array)
        predicted_class = predict_image(img, trained_model)
        print(f'【书法风格识别结果】: {class_dict[predicted_class]}')
        return True
 
 
# 对外调用的训练方法
def train_style(train_path,csv_path,model_path):
    get_train_data_path(
        train_path,csv_path,model_path

    )
    train_and_save_mode()
 
# if __name__ == '__main__':
#     train_and_save_mode()