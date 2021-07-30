from sentence_transformers import SentenceTransformer
import numpy as np
from tensorflow import keras

## load pre-trained model 

model = keras.models.load_model('model/')

q1 = input("Enter your first question ::  ")
q2 = input("Enter your second question ::  ")

is_gpu = input("Do you have GPU on your device with CUDA ? [y/n]\n")
is_gpu = (is_gpu.lower()=='y' or is_gpu.lower()=='yes' )


## Start downloading the transformer model for questions predictions.
if is_gpu:
  emb_model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cuda') ## Use GPU to accelerate the model encoding
else:
  emb_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
  
#Compute embedding for both questions together
embeds = emb_model.encode([q1,q2], convert_to_tensor=True)

model_input = embeds[0]*embeds[1]
model_input = np.array(model_input.tolist())

predictions = model.predict_classes(model_input.reshape((1,model_input.shape[0])))

print("\n\nDuplicated" if (predictions[0][0]==1)  else "\n\nNot Duplicated")
