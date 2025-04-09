import csv
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import os
import random
 
# 设置超参数
num_epochs = 10
batch_size = 32
learning_rate = 0.001
num_classes_per_model = 25  # 每个模型处理的类别数量
num_models = 4  # 总模型数量
threshold = 0.90  # 多模型阈值期望
num_img_per_class = 400
csv_file = 'class_mapping.csv'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
 
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 图像转换
transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ])
 
 
# 清理临时目录
def clean_temp_directories(data_dir, num_models):
    for i in range(num_models):
        temp_dir = os.path.join(data_dir, f'temp_model_{i}')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Temporary directory {temp_dir} has been deleted.")
        else:
            print(f"Temporary directory {temp_dir} does not exist.")
 
 
# 创建全部分类csv标签映射文件
def create_csv_file(all_classes):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for idx, class_name in enumerate(all_classes):
            writer.writerow([idx, class_name])
    print(f'Class mapping saved to {csv_file}')
 
 
# 读取全局的CSV文件，获取类别名称和全局下标的映射
def load_class_mapping():
    class_to_idx = {}
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            idx = int(row[0])
            class_name = row[1]
            class_to_idx[class_name] = idx
    return class_to_idx
 
 
# 数据预处理
def get_data_loaders(data_dir, num_classes_per_model, batch_size, temp_dir):
 
    all_classes = sorted(os.listdir(data_dir))  # 所有分类文件夹
    create_csv_file(all_classes)
    # all_class_to_idx = load_class_mapping()
 
    loaders = []
    for i in range(num_models):
        classes = all_classes[i * num_classes_per_model:(i + 1) * num_classes_per_model]
 
        # 创建临时目录，用于只包含这些类别的训练数据
        tep_dir = os.path.join(temp_dir, f'temp_model_{i}')
        os.makedirs(tep_dir, exist_ok=True)
 
        # 将相关类别复制或链接到临时目录
        for cls in classes:
            src_folder = os.path.join(data_dir, cls)
            dst_folder = os.path.join(tep_dir, cls)
            os.symlink(src_folder, dst_folder)  # 使用符号链接以避免复制大量数据
 
        dataset = datasets.ImageFolder(root=temp_dir+f'/temp_model_{i}', transform=transform)
 
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        loaders.append(loader)
 
    return loaders
 
 
# 定义小规模 ResNet 网络
def create_resnet18(num_classes):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
 
 
# 训练和保存模型
def train_and_save_model(train_loader, num_classes, model_save_path, i):
    model = create_resnet18(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    torch.autograd.set_detect_anomaly(True)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
 
        progress_bar = tqdm(train_loader, desc=f"Mobel {i} is Training, Epoch {epoch + 1} - Training")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
 
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
 
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            # 更新损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            progress_bar.set_postfix(loss=running_loss / (len(progress_bar)), accuracy=100. * correct / total)
 
    torch.save(model.state_dict(), model_save_path)
 
 
# 加载模型
def load_model():
    models_list = []
    for i in range(4):
        if not os.path.isfile(f'MyModel/model_part_{i}.pth'):
            return False
        else:
            model = models.resnet18().to(device)
            num_classes = 25
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            state_dict = torch.load(f'MyModel/model_part_{i}.pth')
            model.load_state_dict(state_dict)
            model.eval()  # 设置模型为评估模式
            models_list.append(model)
    return models_list
 
 
# 加载测试数据
def load_test_data(data_dir, num_classes, num_images_per_class, temp_dir):
 
    shutil.rmtree(temp_dir)
 
    all_classes = os.listdir(data_dir)
    selected_classes = sorted(random.sample(all_classes, num_classes))
    all_class_to_idx = load_class_mapping()
 
    for class_name in selected_classes:
        class_dir = os.path.join(data_dir, class_name)
        images = os.listdir(class_dir)
        selected_images = random.sample(images, num_images_per_class)
 
        # 在临时目录中为每个类创建对应的目录
        class_temp_dir = os.path.join(temp_dir, class_name)
        os.makedirs(class_temp_dir)
 
        for img_name in selected_images:
            src_folder = os.path.join(class_dir, img_name)
            dst_folder = os.path.join(class_temp_dir, img_name)
            os.symlink(src_folder, dst_folder)
 
    dataset = datasets.ImageFolder(root=temp_dir, transform=transform)
    index_list = []
    for label in dataset.classes:
        for i in range(num_images_per_class):
            index_list.append(all_class_to_idx[label])
        dataset.class_to_idx[label] = all_class_to_idx[label]
    dataset.targets = index_list
    dataloader = DataLoader(dataset, batch_size=num_images_per_class, shuffle=False)
 
    return dataloader, selected_classes
 
 
# 模拟模型的预测过程
def predict_image(models_list, image):
    max_probs = []
    preds = []
 
    with torch.no_grad():
        for i, model in enumerate(models_list):
            output = model(image)
            probs = torch.nn.functional.softmax(output, dim=1)  # 计算类别概率
            max_prob, pred = torch.max(probs, 1)  # 获取最大概率和对应的类别
            # 全局下标
            pred += i * num_classes_per_model
 
            max_probs.append(max_prob.item())
            preds.append(pred.item())
 
    # 输出调试信息
    # for idx, (prob, pred) in enumerate(zip(max_probs, preds)):
    #     print(f"Model {idx}: Predicted {pred} with probability {prob:.4f}")
 
    # 返回预测概率最大且超过阈值的类别
    best_pred = -1  # 最大概率下标
    best_prob = -1  # 最大概率
    for i, prob in enumerate(max_probs):
        if prob > threshold and prob > best_prob:
            best_prob = prob
            best_pred = preds[i]
 
    return best_pred
 
 
def test_models(models_list, data_loader, selected_classes, class_idx=0):
    total_correct = 0
    total_images = 0
 
    models_list = [model.to(device) for model in models_list]
 
    for images, labels in tqdm(data_loader, desc=f"Testing"):
        class_correct = 0
        class_total = 0
 
        labels = [data_loader.dataset.class_to_idx[selected_classes[class_idx]]] * int(len(data_loader.dataset.targets) / len(data_loader.dataset.classes))
        labels = torch.tensor(labels)
 
        images = images.to(device)
        labels = labels.to(device)
 
        for i in range(len(images)):
            image = images[i].unsqueeze(0)  # 添加 batch 维度
            label = labels[i].item()
 
            # 使用4个模型测试图像
            pred = predict_image(models_list, image)
 
            print(f"True label: {selected_classes[class_idx]} (Index: {label}), Predicted label: {pred}")
 
            if pred == label:
                class_correct += 1
 
            class_total += 1
 
        # 计算并显示该类的正确率
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        print(f"Accuracy for class {selected_classes[class_idx]}: {class_accuracy:.2f}")
 
        class_idx += 1
 
        total_correct += class_correct
        total_images += class_total
 
    # 计算并显示总正确率
    total_accuracy = total_correct / total_images if total_images > 0 else 0
    print(f"\nOverall accuracy: {total_accuracy:.2f}")
 
 
# 训练和测试主流程
def main(data_dir, temp_dir):
    # clean_temp_directories(temp_dir, num_models)
    # loaders = get_data_loaders(data_dir, num_classes_per_model, batch_size, temp_dir)
    #
    # for i, loader in enumerate(loaders):
    #     model_save_path = f'MyModel/model_part_{i}.pth'
    #     train_and_save_model(loader, num_classes_per_model, model_save_path, i)
    #
    # clean_temp_directories(temp_dir, num_models)
 
    models_list = load_model()
 
    data_loader, selected_classes = load_test_data(data_dir, 20, 10, temp_dir)
 
    test_models(models_list, data_loader, selected_classes)
 
    clean_temp_directories(temp_dir, 1)
 
 
# 对外暴露的书法字形识别函数
def train_font():
    # 训练的数据集
    data_dir = 'E:/My Chinese Calligraphy Styles/train'
    # 临时目录，用于存储各个模型的训练数据
    temp_dir = 'E:/My Chinese Calligraphy Styles/temp'
 
    clean_temp_directories(temp_dir, num_models)
    loaders = get_data_loaders(data_dir, num_classes_per_model, batch_size, temp_dir)
 
    for i, loader in enumerate(loaders):
        model_save_path = f'MyModel/model_part_{i}.pth'
        train_and_save_model(loader, num_classes_per_model, model_save_path, i)
 
    clean_temp_directories(temp_dir, num_models)
 
 
# 对外暴露的书法字形预测函数
def book_font_predict(image_array):
    models_list = load_model()
    if not models_list:
        return False
    else:
        models_list = [model.to(device) for model in models_list]
 
        image = Image.fromarray(image_array)
        image = transform(image).unsqueeze(0)
        image = image.to(device)
 
        pred = predict_image(models_list, image)
        if pred == -1:
            print("无法识别")
        else:
            class_dict = load_class_mapping()
            key = next(k for k, v in class_dict.items() if v == pred)
            print(f'【书法字形识别结果】: {key}, index: {pred}')
        return True
 
 
# if __name__ == "__main__":
#     data_dir = 'E:/My Chinese Calligraphy Styles/train'
#     temp_dir = 'E:/My Chinese Calligraphy Styles/temp'
#     main(data_dir, temp_dir)
#     print("训练模型测试执行完成")