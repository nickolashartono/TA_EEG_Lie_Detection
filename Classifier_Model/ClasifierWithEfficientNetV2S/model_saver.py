from model_builder import LitClassifierWithEfficientNetV2S as classifier_64
from model_builder_128 import LitClassifierWithEfficientNetV2S as classifier_128
from model_builder_single import LitClassifierWithEfficientNetV2S as classifier_single
import torch


ckpt_path = 'E:\\Nicko\\TUGAS_AKHIR\\ClasifierWithEfficientNetV2S\\Logs\\runs_16\\Run_1_feature_1,3,6,7_train\classifier-epoch=9849-F1_Score_avg=0.85.ckpt'
save_path = 'E:\\Nicko\\TUGAS_AKHIR\\ClasifierWithEfficientNetV2S\\Saved_weigths\\model7.pt'

model = classifier_128().load_from_checkpoint(checkpoint_path=ckpt_path)

torch.save(model.state_dict(), save_path)