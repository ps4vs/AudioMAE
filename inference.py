from models_vit import vit_base_patch16

model = vit_base_patch16()

for name, param in model.named_parameters():
    print(name, param.shape)