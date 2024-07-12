#Step1: Load the data and separate into train, validation and test data
# Import necessary libraries
# Install datasets, tokenizers library if you've not done so yet (!pip install datasets, tokenizers).
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

os.mkdir("./malaygpt")
os.mkdir("./tokenizer_en")
os.mkdir("./tokenizer_my")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = load_dataset("Helsinki-NLP/opus-100", "en-ms", split='train')
validation_dataset = load_dataset("Helsinki-NLP/opus-100", "en-ms", split='validation')

# limit the number of data in dataset for faster training purpose
raw_train_dataset, rt_to_skip = random_split(train_dataset, [1500,len(train_dataset)-1500])
raw_validation_dataset, vt_to_skip = random_split(validation_dataset, [50,len(validation_dataset)-50])

#Step2: Create tokenizers

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_ds_iterator(raw_train_dataset, lang):
  for data in raw_train_dataset:
    yield data['translation'][lang]

# Create Source Tokenizer - English
tokenizer_en = Tokenizer(BPE(unk_token="[UNK]"))
trainer_en = BpeTrainer(min_frequency=2, special_tokens=["[PAD]","[UNK]","[CLS]", "[SEP]", "[MASK]"])
# We’ll also need to add a pre-tokenizer to split our input into words as without a pre-tokenizer, we might get tokens that overlap several words: for instance we could get a "there is" token since those two words often appear next to each other.
# Using a pre-tokenizer will ensure no token is bigger than a word returned by the pre-tokenizer.
tokenizer_en.pre_tokenizer = Whitespace()
tokenizer_en.train_from_iterator(get_ds_iterator(raw_train_dataset, "en"), trainer=trainer_en)
tokenizer_en.save("./tokenizer_en/tokenizer_en.json")

# Create Target Tokenizer - Malay
tokenizer_my = Tokenizer(BPE(unk_token="[UNK]"))
trainer_my = BpeTrainer(min_frequency=2, special_tokens=["[PAD]","[UNK]","[CLS]", "[SEP]", "[MASK]"])
tokenizer_my.pre_tokenizer = Whitespace()
tokenizer_my.train_from_iterator(get_ds_iterator(raw_train_dataset, "ms"), trainer=trainer_my)
tokenizer_my.save("./tokenizer_my/tokenizer_my.json")

tokenizer_en = Tokenizer.from_file("./tokenizer_en/tokenizer_en.json")
tokenizer_my = Tokenizer.from_file("./tokenizer_my/tokenizer_my.json")

source_vocab_size = tokenizer_en.get_vocab_size()
target_vocab_size = tokenizer_my.get_vocab_size()

# to calculate the max sequence lenth in the entire training dataset for the source and target dataset
max_seq_len_source = 0
max_seq_len_target = 0

for data in raw_train_dataset:
    enc_ids = tokenizer_en.encode(data['translation']['en']).ids
    dec_ids = tokenizer_my.encode(data['translation']['ms']).ids
    max_seq_len_source = max(max_seq_len_source, len(enc_ids))
    max_seq_len_target = max(max_seq_len_target, len(dec_ids))

print(f'max_seqlen_source: {max_seq_len_source}')   #99 - can be different in your case
print(f'max_seqlen_target: {max_seq_len_target}')   #109 - can be different in your case

# to make it standard for our training we'll just take max_seq_len_source and add 20-50 to cover the additional tokens such as PAD, CLS, SEP
max_seq_len = 155

# Step3: Prepare dataset and dataloader

# Transform raw dataset to the encoded dataset that can be processed by the model
class EncodeDataset(Dataset):
    def __init__(self, raw_dataset, max_seq_len):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):

        # fetching the single data for the given index value that consist of both english and malay language.
        raw_text = self.raw_dataset[index]

        # separating text by source and target lanaguage which will be later used for encoding.
        source_text = raw_text['translation']['en']
        target_text = raw_text['translation']['ms']

        # Encoding source text with with english tokenizer and target text with malay tokenizer
        source_text_encoded = tokenizer_en.encode(source_text).ids
        target_text_encoded = tokenizer_my.encode(target_text).ids

        # Convert the CLS, SEP and PAD tokens to their corresponding index id in vocabulary using tokenizer [the id would be same with either tokenizers]
        CLS_ID = torch.tensor([tokenizer_my.token_to_id("[CLS]")], dtype=torch.int64)
        SEP_ID = torch.tensor([tokenizer_my.token_to_id("[SEP]")], dtype=torch.int64)
        PAD_ID = torch.tensor([tokenizer_my.token_to_id("[PAD]")], dtype=torch.int64)

        # To train the model, the sequence lenth of each input should be equal max seq length. Hence additional number of padding will be added to the input sequence if the lenth is not equal to the max seq length.
        num_source_padding = self.max_seq_len - len(source_text_encoded) - 2
        num_target_padding = self.max_seq_len - len(target_text_encoded) - 1

        encoder_padding = torch.tensor([PAD_ID] * num_source_padding, dtype = torch.int64)
        decoder_padding = torch.tensor([PAD_ID] * num_target_padding, dtype = torch.int64)

        # encoder_input has the first token as start of senstence - CLS_ID, followed by source encoding which is then followed by the end of sentence token - SEP.
        # To reach the required max_seq_len, addition PAD token will be added at the end.
        encoder_input = torch.cat([CLS_ID, torch.tensor(source_text_encoded, dtype=torch.int64), SEP_ID, encoder_padding], dim=0)

        # decoder_input has the first token as start of senstence - CLS_ID, followed by target encoding.
        # To reach the required max_seq_len, addition PAD token will be added at the end. There is no end of sentence token - SEP in decoder input.
        decoder_input = torch.cat([CLS_ID, torch.tensor(target_text_encoded, dtype=torch.int64), decoder_padding ], dim=0)

        # target_label is required for the loss calculation during training to compare between the predicted and target label.
        # target_label has the first token as target encoding followed by actual target encoding. There is no start of sentence token - CLS in target label.
        # To reach the required max_seq_len, addition PAD token will be added at the end.
        target_label = torch.cat([torch.tensor(target_text_encoded, dtype=torch.int64),SEP_ID,decoder_padding], dim=0)

        # Since we've added extra padding token with input encoding, we don't want this token to be trained by model.
        # So, we'll use encoder mask to nullify the padding value prior to producing output of self attention in encoder block
        encoder_mask = (encoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int()

        # We don't want any token to get influence the future token during the decoding stage. Hence, Causal mask is being implemented during masked multihead attention to handle this.
        decoder_mask = (decoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0))

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'target_label': target_label,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'source_text': source_text,
            'target_text': target_text
        }

# Causal mask will make sure any token that comes after the current token will be masked meaning the value will be replaced by -infinity that will be converted to zero or neearly zero after softmax operation. Hence the model will just ignore these value or willn't be able to learn anything.
def causal_mask(size):
        # Creating a square matrix of dimensions 'size x size' filled with ones
        mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
        return mask == 0

# create a dataloader to use for model training and validation
train_ds = EncodeDataset(raw_train_dataset, max_seq_len)
val_ds = EncodeDataset(raw_validation_dataset, max_seq_len)

train_dataloader = DataLoader(train_ds, batch_size = 5, shuffle = True)
val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)

# Step 4: Input embedding and positional encoding
import torch
import torch.nn as nn
import math

class EmbeddingLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        # using pytorch models embedding layer to map token id to embeeding vector which has the shape of (vocab_size, d_model)
        # The vocab_size is the vocabulary size of the training data created by tokenizer in step 2
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input):
        # In addition of giving input to the embedding, the extra multiplication by square root of d_model is to normalize the embedding layer output
        embedding_output = self.embedding(input) * math.sqrt(self.d_model)
        return embedding_output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        pe = torch.zeros(max_seq_len, d_model)

        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        # since we're expecting the input sentenses in batches so the extra dimension to cater batch number needs to be added in 0 postion
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input_embdding):
        input_embdding = input_embdding + (self.pe[:, :input_embdding.shape[1], :]).requires_grad_(False)   # to prevent from calculating gradient
        return self.dropout(input_embdding)
    
# Step 5: Multihead Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float):
        super().__init__()
        # Defining dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by number of heads"

        # d_k is the new dimension of each self attention heads
        self.d_k = d_model // num_heads

        # Weight matrix are defined which are all learnable parameters
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, encoder_mask):

        # Please note that we'll be training our model with not just a single sequence but rather batches of sequence, hence we'll include batch_size in the shape
        # query, Key and value are calculated by matrix multiplication of corresponding weights with the input embeddings
        # Change of shape: q(batch_size, seq_len, d_model) @ W_q(d_model, d_model) => query(batch_size, seq_len, d_model) [same goes to key and value]
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)

        # Dividing query, key and value into number of heads, hence new dimenstion will be d_k.
        # Change of shape: query(batch_size, seq_len, d_model) => query(batch_size, seq_len, num_heads, d_k) -> query(batch_size,num_heads, seq_len,d_k) [same goes to key and value]
        query = query.view(query.shape[0], query.shape[1], self.num_heads ,self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads ,self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads ,self.d_k).transpose(1,2)

        # :: SELF ATTENTION BLOCK STARTS ::

        # Attention score is calculated to find the similarity or relation of query with key of itself and all other embedding in the sequence
        #  Change of shape: query(batch_size,num_heads, seq_len,d_k) @ key(batch_size,num_heads, seq_len,d_k) => attention_score(batch_size,num_heads, seq_len,seq_len)
        attention_score = (query @ key.transpose(-2,-1))/math.sqrt(self.d_k)

        # If mask is provided the attention score needs to modify as per the mask value. Refer to the details in point no 4.
        if encoder_mask is not None:
          attention_score.masked_fill_(encoder_mask==0, -1e9)

        # Softmax operation calculates the probability distribution among all the attention scores. This will determine which embedding is more similar to the given query embedding and assign the attention weight accordingly.
        # Change of shape: same as attention_score
        attention_score = attention_score.softmax(dim=-1)

        if self.dropout is not None:
          attention_score = self.dropout(attention_score)

        # Final step of Self attention block is to matrix multiplication of attention_weight with value embedding.
        # Change of shape: attention_score(batch_size,num_heads, seq_len,seq_len) @  value(batch_size,num_heads, seq_len,d_k) => attention_output(batch_size,num_heads, seq_len,d_k)
        attention_output = attention_score @ value

        # :: SELF ATTENTION BLOCK ENDS ::

        # Now, all the heads will be concated back to for a single head
        # Change of shape:attention_output(batch_size,num_heads, seq_len,d_k) => attention_output(batch_size,seq_len,num_heads,d_k) => attention_output(batch_size,seq_len,d_model)
        attention_output = attention_output.transpose(1,2).contiguous().view(attention_output.shape[0], -1, self.num_heads * self.d_k)

        # Finally attention_output is matrix multiplied with output weight matrix to give the final Multi-Head attention output.
        # The shape of the multihead_output is same as the embedding input
        # Change of shape: attention_output(batch_size,seq_len,d_model) @ W_o(d_model, d_model) => multihead_output(batch_size, seq_len, d_model)
        multihead_output = self.W_o(attention_output)

        return multihead_output    
    
# Step 6: Feedfoward Network, Layer Normalization and AddAndNorm

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_1 = nn.Linear(d_model, d_ff)
        self.layer_2 = nn.Linear(d_ff, d_model)

    def forward(self, input):
        return self.layer_2(self.dropout(torch.relu(self.layer_1(input))))

class LayerNorm(nn.Module):
    # def __init__(self, features:int=512, eps: float = 1e-5):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        # epsilon is a very small value and is plays an important role to avoid division by zero problem
        self.eps = eps
        #Extra learning parameters gamma and beta are introduced to scale and shift the embedding value as the network needed.
        self.gamma = nn.Parameter(torch.ones(512))  # 512 = advisable to initialize with same number as d_model
        self.beta = nn.Parameter(torch.zeros(512))

    def forward(self, input):
        mean = input.mean(dim = -1, keepdim=True)
        std = input.std(dim = -1, keepdim=True)
        return self.gamma * (input - mean)/(std + self.eps) + self.beta

class AddAndNorm(nn.Module):
  def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm()

  def forward(self, input, sub_layer):
        return input + self.dropout(sub_layer(self.layer_norm(input)))

#Step 7: Encoder block and Encoder

class EncoderBlock(nn.Module):
    # def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout_rate: float) -> None:
    def __init__(self, multihead_attention: MultiHeadAttention, feed_forward: FeedForward, dropout_rate: float) -> None:
        super().__init__()
        self.multihead_attention = multihead_attention
        self.feed_forward = feed_forward
        self.addnorm_1 = AddAndNorm(dropout_rate)
        self.addnorm_2 = AddAndNorm(dropout_rate)

    def forward(self, encoder_input, encoder_mask):
        # First AddAndNorm unit taking encoder input from skip connection and adding it with the output of MultiHead attention block
        encoder_input = self.addnorm_1(encoder_input, lambda encoder_input: self.multihead_attention(encoder_input, encoder_input, encoder_input, encoder_mask))
        # Second AddAndNorm unit taking output of MultiHead attention block from skip connection and adding it with the output of Feedforward layer
        encoder_input = self.addnorm_2(encoder_input, self.feed_forward)
        return encoder_input

class Encoder(nn.Module):
    def __init__(self, encoderblocklist: nn.ModuleList) -> None:
        super().__init__()
        # Encoder class initialized by taking encoderblock list
        self.encoderblocklist = encoderblocklist
        self.layer_norm = LayerNorm()

    def forward(self, encoder_input, encoder_mask):
        # Looping through all the encoder block - 6 times
        for encoderblock in self.encoderblocklist:
            encoder_input = encoderblock(encoder_input, encoder_mask)
        # Normalize the final encoder block output and return. This encoder output will be used later on as key and value for the cross attention in decoder block
        encoder_output = self.layer_norm(encoder_input)
        return encoder_output        
    
#Step 8: Decoder block and decoder and the projection

class DecoderBlock(nn.Module):
    # def __init__(self, features: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout_rate: float) -> None:
    def __init__(self, masked_multihead_attention: MultiHeadAttention, cross_multihead_attention: MultiHeadAttention, feed_forward: FeedForward, dropout_rate: float) -> None:
        super().__init__()
        self.masked_multihead_attention = masked_multihead_attention
        self.cross_multihead_attention = cross_multihead_attention
        self.feed_forward = feed_forward
        self.addnorm_1 = AddAndNorm(dropout_rate)
        self.addnorm_2 = AddAndNorm(dropout_rate)
        self.addnorm_3 = AddAndNorm(dropout_rate)

    def forward(self, decoder_input, encoder_output, encoder_mask, decoder_mask):
        # First AddAndNorm unit taking decoder input from skip connection and adding it with the output of Masked Multi-Head attention block
        decoder_input = self.addnorm_1(decoder_input, lambda decoder_input: self.masked_multihead_attention(decoder_input, decoder_input, decoder_input, decoder_mask))
        # Second AddAndNorm unit taking output of Masked Multi-Head attention block from skip connection and adding it with the output of MultiHead attention block
        decoder_input = self.addnorm_2(decoder_input, lambda decoder_input: self.cross_multihead_attention(decoder_input, encoder_output, encoder_output, encoder_mask))
        # Third AddAndNorm unit taking output of MultiHead attention block from skip connection and adding it with the output of Feedforward layer
        decoder_input = self.addnorm_3(decoder_input, self.feed_forward)
        return decoder_input

class Decoder(nn.Module):
    # def __init__(self, features: int, layers: nn.ModuleList) -> None:
    def __init__(self, decoderblocklist: nn.ModuleList) -> None:
        super().__init__()
        self.decoderblocklist = decoderblocklist
        self.layer_norm = LayerNorm()

    def forward(self, decoder_input, encoder_output, encoder_mask, decoder_mask):
        for decoderblock in self.decoderblocklist:
            decoder_input = decoderblock(decoder_input, encoder_output, encoder_mask, decoder_mask)
        decoder_output = self.layer_norm(decoder_input)
        return decoder_output

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.projection_layer = nn.Linear(d_model, vocab_size)

    def forward(self, decoder_output) -> None:
        # Projection layer first take in decoder output and feed into the linear layer of shape (d_model, vocab_size)
        #Change in shape: decoder_output(batch_size, seq_len, d_model) @ linear_layer(d_model, vocab_size) => output(batch_size, seq_len, vocab_size)
        output = self.projection_layer(decoder_output)
        return output    
    
#Step 9: Create and build Transfomer
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, source_embed: EmbeddingLayer, target_embed: EmbeddingLayer, source_pos: PositionalEncoding, target_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()

        self.source_embed = source_embed
        self.source_pos = source_pos
        self.encoder = encoder

        self.target_embed = target_embed
        self.target_pos = target_pos
        self.decoder = decoder

        self.projection_layer = projection_layer

    def encode(self, encoder_input, encoder_mask):
        encoder_input = self.source_embed(encoder_input)
        encoder_input = self.source_pos(encoder_input)
        encoder_output = self.encoder(encoder_input, encoder_mask)
        return encoder_output

    def decode(self, encoder_output, encoder_mask, decoder_input, decoder_mask):
        decoder_input = self.target_embed(decoder_input)
        decoder_input = self.target_pos(decoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask)
        return decoder_output

    def project(self, decoder_output):
        return self.projection_layer(decoder_output)

def build_model(source_vocab_size: int, target_vocab_size: int, source_seq_len: int, target_seq_len: int, d_model: int=512, num_blocks: int=6, num_heads: int=8, dropout_rate: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    source_embed = EmbeddingLayer(d_model, source_vocab_size)
    target_embed = EmbeddingLayer(d_model, target_vocab_size)

    # Create the positional encoding layers
    source_pos = PositionalEncoding(d_model, source_seq_len, dropout_rate)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout_rate)

    # Create the encoder-block-list
    encoderblocklist = []
    for _ in range(num_blocks):
        multihead_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        encoder_block = EncoderBlock(multihead_attention, feed_forward, dropout_rate)
        encoderblocklist.append(encoder_block)
    # Create the encoder
    encoder = Encoder(nn.ModuleList(encoderblocklist))

    # Create the decoder-block-list
    decoderblocklist = []
    for _ in range(num_blocks):
        masked_multihead_attention = MultiHeadAttention(d_model,num_heads, dropout_rate)
        cross_multihead_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        decoder_block = DecoderBlock(masked_multihead_attention, cross_multihead_attention, feed_forward, dropout_rate)
        decoderblocklist.append(decoder_block)
    # Create the decoder
    decoder = Decoder(nn.ModuleList(decoderblocklist))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    # Now that we've initialized all the required blocks of transformer, we can now inititiate a model
    model = Transformer(encoder, decoder, source_embed, target_embed, source_pos, target_pos, projection_layer)

    # For the first time, we'll initialize the model parameters using xavier uniform method. Once training begings the parameters will be updated by the network
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

# Let's build the the final model.
model = build_model(tokenizer_en.get_vocab_size(), tokenizer_my.get_vocab_size(),max_seq_len, max_seq_len, d_model=512).to(device)

# Let's look at the architecture that we've just build ourself
print(model)

#Step 10: Training and Validation of malayGPT

def run_validation(model, validation_ds, tokenizer_en, tokenizer_my, max_seq_len, device, print_msg, global_step):
    model.eval()
    count = 0

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            cls_id = tokenizer_my.token_to_id('[CLS]')
            sep_id = tokenizer_my.token_to_id('[SEP]')

            # Computing the output of the encoder for the source sequence
            encoder_output = model.encode(encoder_input, encoder_mask)
            # for prediction task, the first token that goes in decoder input is the [CLS] token
            decoder_input = torch.empty(1, 1).fill_(cls_id).type_as(encoder_input).to(device)
            # since we need to keep adding the output back to the input until the [SEP] - end token is received.
            while True:
                # check if the max length is received
                if decoder_input.size(1) == max_seq_len:
                    break

                # recreate mask each time the new output is added the decoder input for next token prediction
                decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

                # apply projection only to the next token
                out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

                # apply projection only to the next token
                prob = model.project(out[:, -1])

                # select the token with highest probablity which is a greedy search implementation
                _, next_word = torch.max(prob, dim=1)
                decoder_input = torch.cat(
                    [decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1
                )
                # check if the new token is the end of token
                if next_word == sep_id:
                    break
            # final output is the concatinated decoder input till the end token is reached
            model_out = decoder_input.squeeze(0)

            source_text = batch["source_text"][0]
            target_text = batch["target_text"][0]
            model_out_text = tokenizer_my.decode(model_out.detach().cpu().numpy())

            # Print the source, target and model output
            print_msg('-'*55)
            # print_msg(f"{f'SOURCE: ':>12}{source_text}")
            # print_msg(f"{f'TARGET: ':>12}{target_text}")
            # print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
            print_msg(f'Source Text: {source_text}')
            print_msg(f'Target Text: {target_text}')
            print_msg(f'Predicted by MalayGPT: {model_out_text}')

            if count == 2:
                break

def train_model(preload_epoch=None):
    # The entire training, validation cycle will run for 20 cycles or epochs.
    EPOCHS = 10
    initial_epoch = 0
    global_step = 0

    # Adam is one of the most commonly used optimization algorithms that hold the current state and will update the parameters based on the computed gradients.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-9)

    # If the preload_epoch is not none, that means the training will start with the weights, optimizer that has been last saved and start with preload epoch + 1
    if preload_epoch is not None:
      model_filename = f"./malaygpt/model_{preload_epoch}.pt"
      state = torch.load(model_filename)
      model.load_state_dict(state['model_state_dict'])
      initial_epoch = state['epoch'] + 1
      optimizer.load_state_dict(state['optimizer_state_dict'])
      global_step = state['global_step']

    # The CrossEntropyLoss loss function computes the difference between the projection output and target label.
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_en.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, EPOCHS):
        # torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            target_label = batch['target_label'].to(device) # (B, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            projection_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(projection_output.view(-1, tokenizer_my.get_vocab_size()), target_label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # VALIDATION BLOCK STARTS HERE [Runs every epoch after the training block is complete]
        run_validation(model, val_dataloader, tokenizer_en, tokenizer_my, max_seq_len, device, lambda msg: batch_iterator.write(msg), global_step)

        # Save the model at the end of every epoch
        model_filename = f"./malaygpt/model_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

# Train our model
train_model(preload_epoch=None)

#Step 11: Finally testing our malayGPT model to translated new sentences. Let's give it a try.

def malaygpt(user_input_text):

    # validation using input text
    user_input_text = str(user_input_text).strip()

    # Let's get the model Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_en = Tokenizer.from_file("./tokenizer_en/tokenizer_en.json")
    tokenizer_my = Tokenizer.from_file("./tokenizer_my/tokenizer_my.json")

    # Build our model
    # model = build_model(tokenizer_en.get_vocab_size(), tokenizer_my.get_vocab_size(), max_seq_len, max_seq_len, d_model=512).to(device)
    # model = get_model(tokenizer_en.get_vocab_size(), tokenizer_my.get_vocab_size()).to(device)
    model = build_model(tokenizer_en.get_vocab_size(), tokenizer_my.get_vocab_size(),max_seq_len, max_seq_len, d_model=512).to(device)

    # Load the specific checkpoint of the model that you've saved during training.
    checkpoint_number = 9    # for this test, I am taking checkpoint number 10
    model_filename = f"./malaygpt/model_{checkpoint_number}.pt"
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    # Lets beging the inferencing
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source_text_encoding = tokenizer_en.encode(user_input_text)
        source_text_encoding = torch.cat([
            torch.tensor([tokenizer_en.token_to_id('[CLS]')], dtype=torch.int64),
            torch.tensor(source_text_encoding.ids, dtype=torch.int64),
            torch.tensor([tokenizer_en.token_to_id('[SEP]')], dtype=torch.int64),
            torch.tensor([tokenizer_en.token_to_id('[PAD]')] * (max_seq_len - len(source_text_encoding.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source_mask = (source_text_encoding != tokenizer_en.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        encoder_output = model.encode(source_text_encoding, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(tokenizer_my.token_to_id('[CLS]')).type_as(source_text_encoding).to(device)

        # Generate the translation word by word
        while decoder_input.size(1) < max_seq_len:
            # build mask for target and calculate output
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source_text_encoding).fill_(next_word.item()).to(device)], dim=1)

            # print the translated word
            print(f"{tokenizer_my.decode([next_word.item()])}", end=' ')

            # break if we predict the end of sentence token
            if next_word == tokenizer_my.token_to_id('[SEP]'):
                break

    # convert ids to tokens
    return tokenizer_my.decode(decoder_input[0].tolist())    