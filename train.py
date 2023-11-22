import os
import torch
import data_setup, engine, model_builder, utils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt 
from torchvision import transforms
import pandas as pd
# Setup hyperparametrs 
NUM_EPOCHS = 100
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# set up directories

train_dir =r'C:\Users\19175136\Documents\going_modular\data\pizza_steak_sushi\train'
test_dir = r'C:\Users\19175136\Documents\going_modular\data\pizza_steak_sushi\test'

device = "cuda" if torch.cuda.is_available() else "cpu"

#print(results)
def accuracy_polt(results):
    plt.plot(results['train_acc'])
    plt.plot(results['test_acc'])
    plt.plot(results['train_loss'])
    plt.plot(results['test_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'test_acc', 'train_loss','test_loss'], loc='upper left')
    #plt.show()
    return plt.savefig('model_accuracy.png')

if __name__ == '__main__':
    # create transform
    data_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
        ])

# create Dataloader with the help of data_setup.py

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE)
# create a model with the help of the model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3, 
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), 
                                lr = LEARNING_RATE)

    results = engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)
    pd.DataFrame(results).to_csv("Results", sep='\t')
    accuracy_polt(results)
       
        
    
    utils.save_model(model=model,
                    target_dir=r'C:\Users\19175136\Documents\going_modular\models',
                    model_name="05_going_modular_scripts_model_tinyvgg_model.pth")
