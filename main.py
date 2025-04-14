from config import Preprocessing_Config, Train_Config
from preprocessing import preprocess
from model.facial_image_encoder import VCCT_encoder
from model.transformer.models.embedding.positional_encoding import PositionalEncoding
# from model.FER_ViTnCNN.Models.Checkpoints import 

def get_image_encoder():
    model = VCCT_encoder()
    model.load_state_dict("./model/FER_ViTnCNN/Models/Checkpoints/MCCT-7_OS_ckpt_10.pt")
    return model

def get_positional_encoder():
    return positional_encoder

if __name__ == "__main__":
    preprocess_config = Preprocessing_Config().pilot_test()
    if preprocess_config.preprocess:
        preprocess(preprocess_config)
    
    
    
    
