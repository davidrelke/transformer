import torch
from torch import nn, Tensor


class EmbeddingModule(nn.Module):
    """
    Simple learned lookup for words
    """

    def __init__(self, vocabulary_size: int, model_dimensions: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=model_dimensions)

    def forward(self, x) -> Tensor:
        return self.embedding(x)


class PositionalEncoderModule(nn.Module):
    """
    Learned Positional Encodings
    Original Paper uses fixed encodings based on sin/cos functions, but learned should be ok too.
    """

    def __init__(self, max_input_length: int, model_dimensions: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=max_input_length, embedding_dim=model_dimensions)

    def forward(self, x) -> Tensor:
        return self.embedding(x)


class FeedForwardModule(nn.Module):
    """
    Feed Forward Module for the Transformer Block
    Input (from Multi-head Attention) -> Linear -> ReLu -> Dropout -> Linear
    """

    def __init__(self, model_dimensions: int, forward_expansion: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_features=model_dimensions, out_features=model_dimensions * forward_expansion)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(in_features=model_dimensions * forward_expansion, out_features=model_dimensions)
        self.relu = nn.ReLU()

    def forward(self, x) -> Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, model_dimensions, number_of_attention_heads: int, dropout_rate: float, forward_expansion: int):
        super().__init__()
        # self.attention = MultiHeadAttentionModule()
        self.feed_forward = FeedForwardModule(model_dimensions=model_dimensions, forward_expansion=forward_expansion, dropout_rate=dropout_rate)
        self.norm_1 = nn.LayerNorm(normalized_shape=model_dimensions)
        self.norm_2 = nn.LayerNorm(normalized_shape=model_dimensions)
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.dropout_2 = nn.Dropout(p=dropout_rate)

    def forward(self, input):
        # We apply dropout [33 ] to the output of each sub-layer, before it is added to the sub-layer input and normalized.
        # Attention -> Dropout -> Residual -> Norm -> FF(Linear -> ReLu - > Dropout -> Linear) -> Dropout -> Residual -> Norm

        # Multi Head Attention
        # attention = self.attention(x)
        attention = None
        x = self.dropout_1(attention)  # First Dropout
        x = input + x  # First Residual
        attention_result = self.norm_1(x)  # First Normalization, Save result for residual

        # FF
        x = self.feed_forward(attention_result)  # Feed Forward (Linear -> ReLu - > Dropout -> Linear)
        x = self.dropout_2(x)  # Second Dropout
        x = attention_result + x  # Second Residual
        x = self.norm_2(x)  # Second Normalization

        return x


class EncoderModule(nn.Module):
    """
    The whole Encoder
    Consists of Input embedding and multiple Transformer Blocks
    """

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
