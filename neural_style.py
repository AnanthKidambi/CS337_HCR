import torch
import torchvision.models  as models
import torchvision.transforms as transforms
from torchvision.io import read_image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = models.VGG19_Weights.DEFAULT
model = models.vgg19(weights=weights, progress=True)
model = model.to(device)
model.eval()

print("Device being used:", device)

preprocess = weights.transforms(antialias=True)

img = read_image('input/tubingen.jpg')

img = img.to(device)

layer_names = [f"conv{i}_2" for i in range(1, 6)]
layer_nums = [3, 8, 13, 22, 31] # taken from https://www.mathworks.com/help/deeplearning/ref/vgg19.html

truncated_models = [torch.nn.Sequential(*list(model.features.children())[:layer_num]) for layer_num in layer_nums]

for i in layer_nums:
    assert model.features[i-1]._get_name() == "Conv2d"
    # assert model.features[i]._get_name() == "ReLU"

actual_outputs = []
for truncated_model in truncated_models:
    #change the max pooling layers to avg pooling layers
    for i in range(len(truncated_model)):
        if truncated_model[i]._get_name() == "MaxPool2d":
            truncated_model[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
    actual_outputs.append(truncated_model(preprocess(img).unsqueeze(0)))

pre_means = [0.485, 0.456, 0.406]
pre_stds = [0.229, 0.224, 0.225]

for i, truncated_model in enumerate(truncated_models):
    # create a random gaussian noise image  
    # hyperparams:
    # conv1_2 : epochs-1000, lr-0.05
    # conv2_2 : epochs-500, lr-0.05
    # conv3_2 : epochs-500, lr-0.05
    # if i != 1:
    #   continue
    noise_img = torch.ones([3, 224, 224], device=device) - torch.rand([3, 224, 224], device = device)*0.5
    preprocess(noise_img.unsqueeze(0))
    noise_img.requires_grad = True
    num_steps = 500
    optimizer = torch.optim.LBFGS([noise_img], max_iter=num_steps, tolerance_change=-1, tolerance_grad=-1)
    #run the lbfgs optimizer for 1000 steps
    num_iter = [0]
    iter_range = tqdm(range(num_steps))
    def closure():
        # print("iter: ", num_iter[0])
        iter_range.update(1)
        num_iter[0] += 1
        optimizer.zero_grad()
        output = truncated_model(noise_img.unsqueeze(0))
        # print(output.shape)
        # print(actual_outputs[i].shape)
        loss = 0.5*((output - actual_outputs[i])**2).sum()
        loss.backward(retain_graph=True)
        # print(loss - 0.001*output[output < 0].sum())
        return loss 
    for _ in range(1):
        optimizer.step(closure)
    # optimizer = torch.optim.Adam([noise_img], lr=0.05)
    # for j in tqdm(range(num_steps)):
    #     output = truncated_model(noise_img.unsqueeze(0))
    #     print(output.shape)
    #     print(actual_outputs[i].shape)
    #     dummy_loss = 0.5*((output - actual_outputs[i])**2).sum()
    #     optimizer.zero_grad()
    #     dummy_loss.backward(retain_graph=True)
    #     optimizer.step()

    corr_img = noise_img.clone()
    corr_img[0] = (corr_img[0]*pre_stds[0]) + pre_means[0]
    corr_img[1] = (corr_img[1]*pre_stds[1]) + pre_means[1]
    corr_img[2] = (corr_img[2]*pre_stds[2]) + pre_means[2]
    for i in range(3):
        print(corr_img[i].min(), corr_img[i].max())
        # if corr_img[i].min() < 0:
        #     corr_img[i] -= corr_img[i].min()

    corr_img = transforms.ToPILImage()(corr_img.detach().cpu())
    corr_img.save(f"output/{layer_names[i]}_corr.jpg")