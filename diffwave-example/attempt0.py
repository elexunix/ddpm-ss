import torch, diffwave
#from diffwave.inference import 

model_dir = '.'
model = torch.load('diffwave-ljspeech-22kHz-1000578.pt')['model']
# it is a dict :(
# I have the class, but no hyperparameters to initialize it from the checkpoint
audio, sr = model(model_dir=model_dir, fast_sampling=True)
print(audio, sr)
