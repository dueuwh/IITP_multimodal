from config import Preprocessing_Config, Train_Config
from preprocessing import preprocess
from model.facial_image_encoder import VCCT_encoder
from model.transformer.models.embedding.positional_encoding import PositionalEncoding
# from model.FER_ViTnCNN.Models.Checkpoints import 

if __name__ == "__main__":
    preprocess_config = Preprocessing_Config().MBE()
    if preprocess_config.preprocess:
        preprocess(preprocess_config)
    
    
    
    
