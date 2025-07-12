# from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from sentence_transformers.models import Pooling
from sentence_transformers import models
from sentence_transformers.similarity_functions import SimilarityFunction
import pdb
from info_nce import InfoNCE, info_nce

sentence_grads = [None]
sentence_embeddings = [None]

# sim_model = SentenceTransformer()

def sent_forward_hook(module, input, output):
    sentence_embedding = output['sentence_embedding']
    sentence_embeddings[0] = sentence_embedding.detach()
    # print(grad)
    sentence_embedding.register_hook(lambda grad: sentence_grads.__setitem__(0, grad)) #sentence_grads.__setitem__(0, grad) print(grad)

    
class CustomSentenceEmbedderWithGrad(torch.nn.Module):
    
    def __init__(self, model_name = 'sentence-transformers/sentence-t5-large'):
        # 'sentence-transformers/all-MiniLM-L6-v2'  **also see loss** increase number of generations
        super().__init__()
        # Initialize SentenceTransformer model
        # self.sentence_transformer = SentenceTransformer(model_name).to('cuda:0')
        
        # Access the underlying transformer model
        self.sim_fn = SimilarityFunction.to_similarity_fn("cosine")
        # self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model = models.Transformer(model_name)
        self.pooling = Pooling(word_embedding_dimension = 768)  # Access the pooling layer
        
        # for name, param in self.model.named_parameters():
            # param.requires_grad = False

    def forward(self, sentences):
        # Tokenize and encode without torch.no_grad()
        inputs = self.model.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to('cuda:0')
        # features = {key: val.to('cuda:0') for key, val in inputs.items()}
        # Get word embeddings with gradients enabled
        # pdb.set_trace()
        outputs = self.model.forward(inputs)
        # print(outputs)
        
        # word_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Use the pooling layer from SentenceTransformer to get sentence embeddings
        # pdb.set_trace()
        sentence_embeddings = self.pooling(outputs)
        
        return sentence_embeddings
        


# input_seq is a single string, additional_seq is list of string
def sentence_importance(input_seq, additional_seq, model, semantic_pair_matrix):
    # print(model[1].children())
    embedder = CustomSentenceEmbedderWithGrad(model_name=model).to('cuda')  # pass model name if needed
    handle = embedder.pooling.register_forward_hook(sent_forward_hook)
    
    # Tokenize input sequences
    input_ = [input_seq] + additional_seq

    # Perform manual forward pass without torch.no_grad()
    # embedder.zero_grad()
    out_features = embedder.forward(input_)
    
    # Retrieve the sentence embeddings
    embeddings = out_features['sentence_embedding']  # Sentence embeddings after pooling
    similarities = embedder.sim_fn(embeddings, embeddings)
    
    # Set up the target similarities
    # target_similarities = torch.ones_like(similarities, device='cuda:0')
    target_similarities = semantic_pair_matrix
    # print('sim:', target_similarities)
    # Calculate the loss and backpropagate
    criterion = InfoNCE()
    loss = criterion(similarities, target_similarities)
    # print('loss:',loss)
    # sentence_grads = []  # To store gradients
    loss.backward()
    
    # Remove the hook
    # handle.remove()
    # Calculate sentence importance
    # pdb.set_trace()
    
    with torch.no_grad():
        if sentence_grads[0].shape == sentence_embeddings[0].shape:
            sentence_attr = sentence_grads[0] * sentence_embeddings[0]
        else:
            print('shape mismatch')
            return None

        # Compute the norm and normalize
        sentence_attr = torch.norm(sentence_attr, dim=1)
        sentence_attr = F.normalize(sentence_attr, p=1, dim=0)
    
    return sentence_attr
