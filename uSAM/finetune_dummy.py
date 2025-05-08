"""
Finetunning with micro_SAM
Adapted from: 
https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/sam_finetuning.ipynb
https://github.com/computational-cell-analytics/micro-sam/tree/master/examples/finetuning
To create Dataloaders -> 
https://github.com/constantinpape/torch-em/blob/main/notebooks/tutorial_create_dataloaders.ipynb

Last Modification: March 17 2025
@author: Lupe Villegas, lvillegas@iib.uam.es
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import imageio.v3 as imageio
import sys
import json 
from datetime import datetime


# import torch
# from torch_em.data import MinInstanceSampler
# import micro_sam.training as sam_training



def main():
    """
    Finetune a microSAM model
    
    It supports image data in '.tif' format (multiple slices). 
    Do not use tif stacks
    image folder: contain all original images (8bits)
    segmentation folder: contain all segmented images (32bits)
    
    train_roi: images for train
    val_roi:images for validation
    
    model_type : model_type to initialize the weights that are finetuned.
    checkpoint_name : Checkpoints will be stored in './checkpoints/<checkpoint_name>'

    batch_size: samples per batch to load (default: 1)
    patch_shape : the size of patches for training (default: image size)
    n_objects_per_batch :the number of objects per batch that will be sampled
    device : the device used for training
    n_epochs : how long we train (in epochs)
    """
    

    # SAVEfolder
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(BASE_DIR, "outputs")
    os.makedirs(save_folder, exist_ok = True)
    save_path = os.path.join(save_folder, "data.json")

    if len(sys.argv) < 3:
        print("Missing arguments")
        sys.exit(1)

    ##save in json
    #root_dir = sys.argv[1]
    model = sys.argv[1]
    # minimal_size = int(sys.argv[3])
    # train_percentage = float(sys.argv[4])
    # batch = int(sys.argv[5])
    # epoch = int(sys.argv[6])
    # model_name = sys.argv[7]

    data = {
        "num_images" : model,
        #"img_size" : img_size,
        #"num_train": num_img_train,
        #"Total runtime:" : total_time,
        "Model_name" : model
    }
    with open(save_path, "w") as f:
        json.dump(data,f)
    


    #root_dir = r'C:\Users\malieva_lab\Documents\1Data\Other\ProcessedData\gliomas_fromSandra_feb2025\4237M03_BTIC_12_Scan_001\4237M03_BTIC_12_Tile1_Crop1\4237M03_crop1_imagesTO_microsam\data'

    # #Dataset
    # image_dir = os.path.join(root_dir,'image')  
    # segmentation_dir = os.path.join(root_dir, 'segmentation')
    # num_images = 0

    #     # SAVEfolder
    # save_folder = os.path.join(root_dir, "outputs")
    # os.makedirs(save_folder, exist_ok=True)  
    # save_path = os.path.join(save_folder, f"data_{model_name}.json")
    



    # try:
    #     for p in os.listdir(image_dir):
    #         path_img = os.path.isfile(os.path.join(image_dir, p))
    #         if path_img:
    #             num_images += 1
    #             image = imageio.imread(path_img)
                
    # except: 
    #     print("Same number of images in image and segmentation folders")

    # img_size = image.shape
    # print(num_images)
    # print(img_size)
    # #image_paths = sorted(glob(os.path.join(image_dir, "*")))
    # #segmentation_paths = sorted(glob(os.path.join(segmentation_dir, "*")))
    # #patch_shape

    
    # # Select the images for the train and the other frames for the validation
    # #get number of images. 
    
    # num_img_train = round(train_percentage*num_images/100) 
    # train_roi = np.s_[:num_img_train , :, :] 
    # val_roi = np.s_[num_img_train:, :, :]
    # print(num_img_train)
        
    # model_type = model 
    # checkpoint_name = model_name #"microsam_gliomas_10epochb "

    # # Train an additional convolutional decoder for end-to-end automatic instance segmentation
    # train_instance_segmentation = True
    # # The sampler chosen below makes sure that the chosen inputs have at least one foreground instance, and filters out small objects.
    # sampler = MinInstanceSampler(min_size = minimal_size)  # NOTE: The choice of 'min_size' value is paired with the same value in 'min_size' filter in 'label_transform'.
    
    # # All hyperparameters for training.
    # batch_size = batch
    
    # patch_shape = (1, 1000, 1000)  
    # n_objects_per_batch = 1 
    # device = torch.device("cuda")  
    # #device = "cuda" if torch.cuda.is_available() else "cpu" # the device/GPU used for training
    # n_epochs =  epochs
    
    # star_time = datetime.now()
    # print ("Start:",star_time)
    
    # for count in range(9):
    #     #print(time.ctime())
    #      # Prints the current time with a five second difference
    #     datetime.sleep(5)
    
    # end_time = datetime.now()
    # total_time = end_time - star_time
    # print ("Total runtime:",total_time)

    # # SAVEfolder
    # save_folder = os.path.join(root_dir, "outputs")
    # os.makedirs(save_folder, exist_ok=True)  
    # save_path = os.path.join(save_folder, f"data_{model_name}.json")
    


    # ##save in json
    # import json 
    # data = {
    #     "num_images" : num_images,
    #     "img_size" : img_size,
    #     "num_train": num_img_train,
    #     "Total runtime:" : total_time,
    #     "Model_name" : model_name
    # }
    # with open(save_path, "w") as f:
    #     json.dump(data,f)




if __name__ == "__main__":

    # root_dir = sys.argv[1]
    # model = sys.argv[2]
    # minimal_size = int(sys.argv[3])
    # train_percentage = float(sys.argv[4])
    # batch = int(sys.argv[5])
    # epoch = int(sys.argv[6])
    # model_name = sys.argv[7]
    # main(root_dir, model, minimal_size, train_percentage, batch, epoch, model_name)
    main()
    
    

