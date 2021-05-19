from torchvision import models, datasets, transforms
import torch
from collections import OrderedDict
import argparse
import json
from torch import nn
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type = str, default = 'flowers/test/11/image_03098.jpg', help = 'path to flowers images') 
parser.add_argument('--checkpoint', type = str, default = 'checkpointfolder/checkfile.pth', help = 'trained model') 
parser.add_argument('--top_k', type = int, default = 3, help = 'Top_k most likely classes') 
parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'mapping of categories to real names') 
parser.add_argument('--gpu', type = bool, default = True, help = 'Utilizes gpu for model processing')

in_args = parser.parse_args() 

image = in_args.data_dir

with open(in_args.category_names, 'r') as f:
    cat_to_name = json.load(f)

model_attr=torch.load(in_args.checkpoint)
    
model = models.vgg13(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.class_to_idx = model_attr['class_to_idx']
hidden_units_size = model_attr['hidden_units_size']
    
classifier  = nn.Sequential(OrderedDict([
                        ('fc1',nn.Linear(25088,hidden_units_size)),
                        ('relu1',nn.ReLU()),
                        ('drop',nn.Dropout(0.2)),
                        ('fc3',nn.Linear(hidden_units_size,512)),
                        ('relu3',nn.ReLU()),
                        ('drop3',nn.Dropout(0.2)),
                        ('fc4',nn.Linear(512,102)),
                        ('softmax',nn.LogSoftmax(dim=1))
                        ]))

model.classifier = classifier
model.load_state_dict(model_attr['model_state_dict'])
                    
img = Image.open(image)

if img.size[0] > img.size[1]:
    img.thumbnail((10000, 256))
else:
    img.thumbnail((256, 10000))
        
img =img.crop(((img.width-224)/2,(img.height-224)/2,(img.width/2 + 112),(img.height/2 + 112)))
img = np.array(img)/255
mean = np.array([0.485, 0.456, 0.406]);std = np.array([0.229, 0.224, 0.225])
img = (img - mean) / std
img =img.transpose((2,0,1))

img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
model_input_img = img_tensor.unsqueeze_(0)


model.eval()
with torch.no_grad():
    ps = torch.exp(model.forward(model_input_img)) 
    top_prob, top_class = ps.topk(in_args.top_k, dim=1)
    #print(top_prob, top_class)
    #print(top_class.tolist())
    counter=0
    print("***The results of the image classification are:***")
    for i in top_class.tolist()[0]:
        print(f"Probability = {top_prob.tolist()[0][counter]:.3} for {cat_to_name[str(i)]}")
        counter+=1
                          
                    
      