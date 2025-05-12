import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from azureml.core.model import Model

# do not modify this code
device = torch.device("cpu")
input_dim, output_dim = (3, 32, 32), 10
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
model_mapping = {}


#########################################
#### Primary functions for score.py #####
#########################################

def init():
    """
    Load registered models from the directory Model.get_model_path('<model_name>')
    and initialize model_mapping as a mapping from model file name to loaded model.

    Note: <model_name> here just refers to the name you registered the model in step 2.
    
    When this function finishes, model_mapping should look like
    
    {
        "trained_LogisticRegression.pt" : <loaded model from trained_LogisicRegression.pt>,
        "trained_LeNet.pt" : <loaded model from trained_LeNet.pt>,
        "trained_AlexNet.pt" : <loaded model from trained_AlexNet.pt>,
        "tuned_{classname}.pt" : <loaded model from tuned_{classname}.pt>
    }
    
    Here classname should be either "LogisticRegression", "LeNet" or "AlexNet",
    depending on which model you fine-tuned earlier.
    """
    global model_mapping
    
    models_base_path = Model.get_model_path('pytorch_models')
    
    logistic_model_path = models_base_path + "/trained_LogisticRegression.pt"
    leNet_model_path = models_base_path + "/trained_LeNet.pt"
    alexNet_model_path = models_base_path + "/trained_AlexNet.pt"
    alexNet_tuned_model_path = models_base_path + "/tuned_AlexNet.pt"
    
    #Logistic
    logistic_model = LogisticRegression(input_dim, output_dim)
    logistic_model.load_state_dict(torch.load(logistic_model_path, map_location = device))
    
    #LeNet
    lenet_model = LeNet(input_dim, output_dim)
    lenet_model.load_state_dict(torch.load(leNet_model_path, map_location = device))
    
    #AlexNet
    alexnet_model = AlexNet(input_dim, output_dim)
    alexnet_model.load_state_dict(torch.load(alexNet_model_path, map_location = device))
    
    #Tuned AlexNet
    alexnettuned_model = AlexNet(input_dim, output_dim)
    alexnettuned_model.load_state_dict(torch.load(alexNet_tuned_model_path, map_location = device))
    
    model_mapping["trained_LogisticRegression.pt"] = logistic_model
    model_mapping["trained_LeNet.pt"] = lenet_model
    model_mapping["trained_AlexNet.pt"] = alexnet_model
    model_mapping["tuned_AlexNet.pt"] = alexnettuned_model


def run(input_data):
    """
    Get the label prediction and associated probability value of an input image
    from each model in model_mapping.
    
    args:
        data (Dict[str, List[float]]):
            the input JSON, which maps the key "image" to a flattened image (a 3072-element Python list)
    
    return:
        Dict[str, Dict[str, str]]: a dictionary where each key is a model name from model_mapping,
            and each value is an inner dictionary with format {"label" : str, "probability" : str}
    
    """
    # do not modify this code
    input_json = json.loads(input_data)
    image_data = np.asarray(input_json["image"]).reshape(1, 3, 32, 32)
    return get_predictions(image_data, model_mapping)


def get_predictions(data, model_mapping):
    """
    Copy the code from get_predictions (Q11) here
    """
    model_prob_dict = {}
    
    input_data_float = torch.FloatTensor(data)
    input_data_float = input_data_float.to(device)
    
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        for model_name, model_obj in model_mapping.items():
            model_obj.to(device)
            model_obj.eval()
            outputs = model_obj(input_data_float)
            probability = softmax(outputs).cpu().numpy()
            max_prob_idx = np.argmax(probability)
            
            predicted_label = classes[max_prob_idx]
            p=probability[0][max_prob_idx]
            predicted_probabilty = str(round(p, 2))
            
            model_prob_dict[model_name] = {"label":predicted_label, "probability": predicted_probabilty}
            
    return model_prob_dict

#########################################
## Copy of model class implementations ##
#########################################

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Copy the code from __init__ in LogisticRegression here
        """
        super(LogisticRegression, self).__init__()
        input_features = input_dim[0] * input_dim[1] *  input_dim[2]
        self.layers = nn.Linear(input_features, output_dim)
    
    def forward(self, x):
        """
        Copy the code from forward in LogisticRegression here
        """
        x = x.reshape(x.size(0), -1)
        out = self.layers(x)
        return out


class LeNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Copy the code from __init__ in LeNet here
        """
        super(LeNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(5*5*16,120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_dim)
        )

    def forward(self, x):
        """
        Copy the code from forward in LeNet here
        """
        return self.layers(x)


class AlexNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Copy the code from __init__ in AlexNet here
        """
        super(AlexNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), #2
            nn.ReLU(inplace=True), #3
            nn.MaxPool2d(kernel_size=2, stride=2), #4
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), #5
            nn.ReLU(inplace=True), #6
            nn.MaxPool2d(kernel_size=2, stride=2), #7
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1), #8
            nn.ReLU(inplace=True), #9
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), #10
            nn.ReLU(inplace=True), #11
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), #12
            nn.ReLU(inplace=True), #13
            nn.MaxPool2d(kernel_size=2, stride=2), #14
            nn.Dropout(), #15
            nn.Flatten(), 
            nn.Linear(2*2*256,4096), #16
            nn.ReLU(inplace=True), #17
            nn.Dropout(), #18 
            nn.Linear(4096,1024), #19
            nn.ReLU(inplace=True), #20
            nn.Linear(1024, output_dim) #21
        )

    def forward(self, x):
        """
        Copy the code from forward in AlexNet here
        """
        return self.layers(x)
