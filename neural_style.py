from os import access
from sklearn import dummy
from sympy import content
import torch
import torchvision.models  as models
import torchvision.transforms as transforms
from torchvision.io import read_image
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = models.VGG19_Weights.DEFAULT
model = models.vgg19(weights=weights, progress=True)
model = model.to(device)
model.eval()

print("Device being used:", device)

preprocess = weights.transforms(antialias=True)

content_img = read_image('input/tubingen.jpg')
content_img = content_img.to(device)

style_img = read_image('input/vangogh_starry_night.jpg')
style_img = style_img.to(device)

style_layer_nums = [1, 6, 11, 20, 29] # taken from https://www.mathworks.com/help/deeplearning/ref/vgg19.html
content_layer_num = 22

content_layers = {f"features.{content_layer_num}": "relu4_2"}
style_layers = {f"features.{i}": f"relu{j+1}_1" for j, i in enumerate(style_layer_nums)}
# return_layers = {**content_layers, **style_layers}

_style_wt_list = [1e3/n**2 for n in [64,128,256,512,512]]
style_weights = {f"relu{j+1}_1":_style_wt_list[j] for j, _ in enumerate(style_layer_nums)}
content_weights = {"relu4_2" : 1e2}

model_ex_style = create_feature_extractor(model, return_nodes=style_layers)
model_ex_content = create_feature_extractor(model, return_nodes=content_layers)

pre_means = [0.485, 0.456, 0.406]
pre_stds = [0.229, 0.224, 0.225]

actual_style_outputs = model_ex_style(preprocess(style_img).unsqueeze(0))
actual_style_outputs_flat = {key: val.view(val.shape[1], -1) for key, val in actual_style_outputs.items()}
actual_gram_matrices = {key: torch.matmul(val, val.t()) for key, val in actual_style_outputs_flat.items()}

actual_content_outputs = model_ex_content(preprocess(content_img).unsqueeze(0))

# #verify the output by truncating the model
# truncated_model = model.features[:content_layer_num+1]
# truncated_output = truncated_model(preprocess(img).unsqueeze(0))
# assert truncated_model[content_layer_num]._get_name() == "ReLU"
# assert torch.allclose(actual_output["relu4_2"], truncated_output)
# exit()

for i in style_layer_nums:
    assert model.features[i]._get_name() == "ReLU"

# pre_means = [0.485, 0.456, 0.406]
# pre_stds = [0.229, 0.224, 0.225]

#regenerate the image content
def regenrate_content():
    for i in range(1, 1+len(style_layer_nums)):
        noise_img = torch.ones([3, 224, 224], device=device) - torch.rand([3, 224, 224], device = device)*0.5
        noise_img.requires_grad = True
        num_steps = 500
        optimizer = torch.optim.Adam([noise_img], lr=0.05)
        for _ in tqdm(range(num_steps)):
            output = model_ex(noise_img.unsqueeze(0))[f"relu{i}_1"]
            dummy_loss = 0.5*((output - actual_outputs[f"relu{i}_1"])**2).sum()

            optimizer.zero_grad()
            dummy_loss.backward(retain_graph=True)

            optimizer.step()

        corr_img = noise_img.clone()
        corr_img[0] = (corr_img[0]*pre_stds[0]) + pre_means[0]
        corr_img[1] = (corr_img[1]*pre_stds[1]) + pre_means[1]
        corr_img[2] = (corr_img[2]*pre_stds[2]) + pre_means[2]
        
        corr_img = transforms.ToPILImage()(corr_img.cpu())
        corr_img.save(f"output/relu{i}_1_corr.jpg")

# regenerate the image style
def regenerate_style():
    noise_img = torch.rand([3, 224, 224], device = device)*0.5
    noise_img.requires_grad = True
    num_steps = 1000
    optimizer = torch.optim.Adam([noise_img], lr=0.1)
    for j in tqdm(range(num_steps)):
        outputs = model_ex(noise_img.unsqueeze(0))
        loss = 0
        for key, val in outputs.items():
            if key in content_layers.values():
                continue
            output_flat = val.view(val.shape[1], -1)
            gram_matrix = torch.matmul(output_flat, output_flat.t())
            loss += style_weights[key] * ((gram_matrix - actual_gram_matrices[key])**2).sum()/(4*(output_flat.shape[0]**2)*(output_flat.shape[1]**2))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    corr_img = noise_img.clone()
    corr_img[0] = (corr_img[0]*pre_stds[0]) + pre_means[0]
    corr_img[1] = (corr_img[1]*pre_stds[1]) + pre_means[1]
    corr_img[2] = (corr_img[2]*pre_stds[2]) + pre_means[2]

    corr_img = transforms.ToPILImage()(corr_img.cpu())
    corr_img.save(f"output/style.jpg")

#getting the required image
noise_img = preprocess(content_img)
# noise_img = torch.rand([3, 224, 224], device = device)*0.5
noise_img.requires_grad = True
num_steps = 1000
optimizer = torch.optim.Adam([noise_img], lr=0.1)
for j in tqdm(range(num_steps)):
    style_outputs = model_ex_style(noise_img.unsqueeze(0))
    content_outputs = model_ex_content(noise_img.unsqueeze(0))
    loss = 0
    for key, val in style_outputs.items():
        output_flat = val.view(val.shape[1], -1)
        gram_matrix = torch.matmul(output_flat, output_flat.t())
        loss += style_weights[key] * ((gram_matrix - actual_gram_matrices[key])**2).mean()
    for key, val in content_outputs.items():
        # print(loss)
        loss += content_weights[key]*((val - actual_content_outputs[key])**2).mean()
        # print(loss)
        # exit()
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

corr_img = noise_img.clone()
corr_img[0] = (corr_img[0]*pre_stds[0]) + pre_means[0]
corr_img[1] = (corr_img[1]*pre_stds[1]) + pre_means[1]
corr_img[2] = (corr_img[2]*pre_stds[2]) + pre_means[2]

corr_img = transforms.ToPILImage()(corr_img.cpu())
corr_img.save(f"output/final.jpg")