import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.distributions.categorical import Categorical

class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, 
                                      embedding_dim=embed_size, 
                                      padding_idx = dataset.pad_id)
        self.rnn = rnn_type(input_size=embed_size, hidden_size=hidden_size, 
                            num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        indices = indices[:, :max(lengths)]
        output, hidden_size = self.rnn(self.embedding(indices))
        logits = self.linear(output)
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()

        start = [self.dataset.bos_id] + self.dataset.text2ids(prefix)
        tokens = torch.tensor(start).unsqueeze(0)
        output, hidden_size = self.rnn(self.embedding(tokens))
        logits = self.linear(output) / temp
        cur_token = Categorical(logits=logits[:, -1:]).sample()
        previous_and_current = torch.cat([tokens, cur_token], dim=1)

        while previous_and_current.shape[1] < self.max_length and cur_token[0] != self.dataset.eos_id:
            output, hidden_size = self.rnn(self.embedding(cur_token), hidden_size)
            logits = self.linear(output) / temp
            cur_token = Categorical(logits=logits[:, -1:]).sample()
            previous_and_current = torch.cat([previous_and_current, cur_token], dim=1)

        generated = self.dataset.ids2text(previous_and_current[0])
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        return generated
