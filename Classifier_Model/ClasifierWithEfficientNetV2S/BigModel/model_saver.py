from model_builder_big import LitClassifierWithEfficientNetV2S as classifier_big
import torch


ckpt_path = 'E:\\Nicko\\TUGAS_AKHIR\\ClasifierBigModel\\Logs\\runs\\Run_11\\last.ckpt'
save_path = 'E:\\Nicko\\TUGAS_AKHIR\\ClasifierWithEfficientNetV2S\\Saved_weigths\\model_big.pt'

model = classifier_big().load_from_checkpoint(checkpoint_path=ckpt_path)

torch.save(model.state_dict(), save_path)