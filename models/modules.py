import torch
from torch import nn, Tensor


class EmbeddingModule(nn.Module):
    def __init__(self, vocabulary_size: int, model_dimensions: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=model_dimensions)

    def forward(self, x) -> Tensor:
        return self.embedding(x)


class PositionalEncoderModule(nn.Module):
    '''
    Learned Positional Encodings
    Original Paper uses fixed encodings based on sin/cos functions, but learned should be ok too.
    '''
    def __init__(self, max_input_length: int, model_dimensions: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=max_input_length, embedding_dim=model_dimensions)

    def forward(self, x) -> Tensor:
        return self.embedding(x)


class EncoderModule(nn.Module):
    def __init__(self, vocabulary_size: int, model_dimensions: int, number_of_attention_heads: int, number_of_layers: int, max_seq_length: int, device: str):
        super().__init__()
        self.device = device
        self.n_encoder_layers: int = number_of_layers
        self.word_embedder = EmbeddingModule(vocabulary_size=vocabulary_size, model_dimensions=model_dimensions)
        self.positional_encoder = PositionalEncoderModule(max_input_length=max_seq_length, model_dimensions=model_dimensions)
        # create number_of_layers encoder blocks
        # create norm layer

    def forward(self, source, mask):
        number_of_samples, seq_length = source.shape
        # positions: arange: 0123...seq_length -> expand: [0123...seq_length, 0123...seq_length, 0123...seq_length] number_of_samples times
        positions: Tensor = torch.arange(start=0, end=seq_length).expand(number_of_samples, seq_length).to(self.device)

        source_embedded = self.word_embedder(source)
        position_encodings = self.positional_encoder(positions)
        encoder_input = source_embedded + position_encodings

        # for layer in encoder_layers: etc.