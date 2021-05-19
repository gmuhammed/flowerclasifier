from torchvision import models, datasets, transforms
import torch
from collections import OrderedDict
from torch import nn, optim
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type = str, default = 'flowers', help = 'path to the folder of flowers images') 
parser.add_argument('--arch', type = str,default = 'vgg13', help = 'CNN Model') 
parser.add_argument('--checkpoint', type = str, default = 'checkpointfolder', help = 'Trained model') 
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning rate')
parser.add_argument('--hidden_units', type = int, default = 4096, help = 'Hidden units')
parser.add_argument('--epochs', type = int, default = 3, help = 'Epochs') 
parser.add_argument('--gpu', type = bool, default = True, help = 'Utilizes gpu for model processing')



in_args = parser.parse_args()   
       
lr = in_args.learning_rate
epochs = in_args.epochs
hidden_units_size = in_args.hidden_units
pretrained_model = in_args.arch
print(in_args.data_dir)


train_dir = in_args.data_dir + '/train'
valid_dir = in_args.data_dir + '/valid'

train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

train_datasets = datasets.ImageFolder(train_dir,train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir,test_transforms)

trainloaders = torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle=True)
validloaders = torch.utils.data.DataLoader(valid_datasets,batch_size=64,shuffle=False)

model = eval("models." + pretrained_model +"(pretrained = True)")
for p in model.parameters():
    p.requires_grad = False
    
classifier  = nn.Sequential(OrderedDict([
                        ('fc1',nn.Linear(25088,hidden_units_size)),
                        ('relu1',nn.ReLU()),
                        ('drop1',nn.Dropout(0.2)),
                        ('fc3',nn.Linear(hidden_units_size,512)),
                        ('relu3',nn.ReLU()),
                        ('drop3',nn.Dropout(0.2)),
                        ('fc4',nn.Linear(512,102)),
                        ('softmax',nn.LogSoftmax(dim=1))
                        ]))

model.classifier = classifier 

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(),lr)


device_input = in_args.gpu
if device_input:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
        
        
model.to(device)
steps = 0
print_every = 200

for e in range(epochs):
    running_loss = 0
    valid_loss = 0
    
    for images,labels in trainloaders:
        steps += 1
        optimizer.zero_grad()
        images,labels = images.to(device),labels.to(device)
        out = model.forward(images)
        loss = criterion(out,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            accuracy=0
            valid_loss = 0
            with torch.no_grad():
                for images,labels in validloaders:
                    images,labels = images.to(device),labels.to(device)
                    logp = model.forward(images)
                    loss = criterion(logp,labels)
                    valid_loss += loss.item()
                    p = torch.exp(logp)
                    top_ps,top_class = p.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor))
                    
            print(f"Epoch {e+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every :.3f} "
                  f"Validation loss: {valid_loss/len(validloaders):.3f} "
                  f"Validation accuracy: {100 * accuracy/len(validloaders):.3f}")
            running_loss = 0
            model.train()

checkpoint = {
              'epoch':epochs,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'class_to_idx': train_datasets.class_to_idx,
              'pretrained_model': pretrained_model,
              'learning_rate': lr,
              'hidden_units_size': hidden_units_size
                }
torch.save(checkpoint, in_args.checkpoint + "/" + 'checkfile.pth' )














