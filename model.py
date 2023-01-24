#Neural Style transfer

import torch  # for model
import torch.nn as nn
import torch.optim as optim
from PIL import Image  #for importing images
import torchvision.models as models  #to load vgg 19 model
import torchvision.transforms as transforms  #to transform the images
from torchvision.utils import save_image #to save the generated images
from tqdm import tqdm # progress bar


#model = models.vgg19(pretrained = True).features  #model

#print(model)

['0', '5', '10', '19', '28']   #layers we want for the loss function

class VGGNet(nn.Module):

    def __init__(self):

        super(VGGNet, self).__init__()

        self.chosen_features = ['0', '5', '10', '19', '28'] 
        self.vgg = models.vgg19(pretrained = True).features

    
    def forward(self,x):

        features = []

        for layer_num, layer in self.vgg._modules.items():

            x = layer(x)

            if layer_num in self.chosen_features:
                features.append(x)
        
        return features


def load_image(image_name):

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)


crops = ['crop3', 'crop4']

for crop_name in crops:

    print(crop_name)

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")  #loading model and freezing the weights

    image_size = 356

    loader = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]
    )

    original_img = load_image("crop1.jpg")
    style_img = load_image(crop_name + '.jpg')



    model = VGGNet().to(device).eval() 

    #making a copy of the origibal image

    generated_img = original_img.clone().requires_grad_(True)


    #hyperparams

    total_steps = 6000
    learning_rate = 0.001
    alpha = 1             #hyperparameters according to paper, for content loss, to provide how much style is needed in ithe image
    beta = 0.01
    loss = []

    optimizer = optim.Adam([generated_img], lr = learning_rate)

    for step in tqdm(range(total_steps)):

        #first we send the 3 images from the vgg network

        generated_feats = model(generated_img)
        original_image_feats = model(original_img)
        style_feats = model(style_img)

        #defining the style loss

        style_loss = original_loss =  0
        

        for gen_feat, orig_image_feat, styl_feat in zip(generated_feats, original_image_feats, style_feats):

            batch_size, channel, height, width = gen_feat.shape

            original_loss += torch.mean((gen_feat - orig_image_feat)**2)

            # computing gram matrix for gen and style to compute style loss

            G = gen_feat.view(channel, height*width).mm(

                gen_feat.view(channel, height*width).t()
            )

            #some sort of correlation matrix

            A = styl_feat.view(channel, height*width).mm(

                styl_feat.view(channel, height*width).t()
            )

            style_loss += torch.mean((G-A)**2)

        total_loss = alpha*original_loss + beta*style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 500 == 0:
            save_str = crop_name + 'generated_img_' + str(step) + '.png'
            loss.append(total_loss)
            save_image(generated_img, save_str)




        

