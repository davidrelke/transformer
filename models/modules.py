import math

import torch
from torch import nn, Tensor
from torch.nn import Linear, Dropout, Softmax


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

    def  __init__(self, model_dimensions: int, forward_expansion: int, dropout_rate: float = 0.1):
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


class EncoderLayer(nn.Module):
    def __init__(self, model_dimensions, number_of_attention_heads: int, dropout_rate: float, forward_expansion: int):
        super().__init__()
        self.attention = MultiHeadSelfAttentionModule(model_dimensions=model_dimensions, dropout_rate=dropout_rate, number_of_heads=number_of_attention_heads)
        self.feed_forward = FeedForwardModule(model_dimensions=model_dimensions, forward_expansion=forward_expansion, dropout_rate=dropout_rate)
        self.norm_1 = nn.LayerNorm(normalized_shape=model_dimensions)
        self.norm_2 = nn.LayerNorm(normalized_shape=model_dimensions)
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.dropout_2 = nn.Dropout(p=dropout_rate)

    def forward(self, input, mask):
        # We apply dropout [33 ] to the output of each sub-layer, before it is added to the sub-layer input and normalized.
        # Attention -> Dropout -> Residual -> Norm -> FF(Linear -> ReLu - > Dropout -> Linear) -> Dropout -> Residual -> Norm

        attention = self.attention(input, input, input, mask)
        x = self.dropout_1(attention)  # First Dropout
        x = input + x  # First Residual
        attention_result = self.norm_1(x)  # First Normalization, Save result for residual

        # FF
        x = self.feed_forward(attention_result)  # Feed Forward (Linear -> ReLu - > Dropout -> Linear)
        x = self.dropout_2(x)  # Second Dropout
        x = attention_result + x  # Second Residual
        x = self.norm_2(x)  # Second Normalization

        return x

class DecoderLayer(nn.Module):
    def __init__(self, model_dimensions, number_of_attention_heads: int, dropout_rate: float, forward_expansion: int):
        super().__init__()
        self.attention_1 = MultiHeadSelfAttentionModule(model_dimensions=model_dimensions, dropout_rate=dropout_rate, number_of_heads=number_of_attention_heads)
        self.attention_2 = MultiHeadSelfAttentionModule(model_dimensions=model_dimensions, dropout_rate=dropout_rate, number_of_heads=number_of_attention_heads)
        self.feed_forward = FeedForwardModule(model_dimensions=model_dimensions, forward_expansion=forward_expansion, dropout_rate=dropout_rate)
        self.norm_1 = nn.LayerNorm(normalized_shape=model_dimensions)
        self.norm_2 = nn.LayerNorm(normalized_shape=model_dimensions)
        self.norm_3 = nn.LayerNorm(normalized_shape=model_dimensions)
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.dropout_2 = nn.Dropout(p=dropout_rate)
        self.dropout_3 = nn.Dropout(p=dropout_rate)

    def forward(self, input, value_from_encoder, key_from_encoder, src_mask, target_mask):
        # We apply dropout [33 ] to the output of each sub-layer, before it is added to the sub-layer input and normalized.
        # Masked Attention -> Dropout -> Residual -> Norm ->
        # Attention (From Encoder) -> Dropout -> Residual -> Norm ->
        # FF(Linear -> ReLu - > Dropout -> Linear) -> Dropout -> Residual -> Norm

        attention = self.attention_1(input, input, input, target_mask)
        x = self.dropout_1(attention)  # First Dropout
        x = input + x  # First Residual
        attention_result = self.norm_1(x)  # First Normalization, Save result for residual

        attention = self.attention_2(value_from_encoder, key_from_encoder, attention_result, src_mask)
        x = self.dropout_2(attention)
        x = attention_result + x
        attention_result = self.norm_2(x)

        # FF
        x = self.feed_forward(attention_result)  # Feed Forward (Linear -> ReLu - > Dropout -> Linear)
        x = self.dropout_3(x)  # Third Dropout
        x = attention_result + x  # Third Residual
        x = self.norm_3(x)  # Third Normalization

        return x


class EncoderModule(nn.Module):
    """
    The whole Encoder
    Consists of Input embedding and multiple Encoder Layers
    """

    def __init__(self, vocabulary_size: int, model_dimensions: int, number_of_attention_heads: int, number_of_layers: int, max_seq_length: int, forward_expansion: int, device: str):
        super().__init__()
        self.device = device
        self.n_encoder_layers: int = number_of_layers
        self.word_embedder = EmbeddingModule(vocabulary_size=vocabulary_size, model_dimensions=model_dimensions)
        self.positional_encoder = PositionalEncoderModule(max_input_length=max_seq_length, model_dimensions=model_dimensions)

        encoder_blocks = []
        for i in range(number_of_layers):
            encoder_blocks.append(EncoderLayer(model_dimensions=model_dimensions, number_of_attention_heads=number_of_attention_heads, forward_expansion=forward_expansion))

        self.encoder_layers = nn.Sequential(*encoder_blocks)

        self.norm = NormModule(model_dimensions=model_dimensions)
        # create norm layer

    def forward(self, source, mask):
        number_of_samples, seq_length = source.shape
        # positions: arange: 0123...seq_length -> expand: [0123...seq_length, 0123...seq_length, 0123...seq_length] number_of_samples times
        positions: Tensor = torch.arange(start=0, end=seq_length).expand(number_of_samples, seq_length).to(self.device)

        source_embedded = self.word_embedder(source)
        position_encodings = self.positional_encoder(positions)
        encoder_input = source_embedded + position_encodings
        # Dropout?

        x = self.encoder_layers(encoder_input, mask)
        x = self.norm(x)
        return x

class DecoderModule(nn.Module):
    def __init__(self, vocabulary_size: int, model_dimensions: int, number_of_attention_heads: int, number_of_layers: int, max_seq_length: int, forward_expansion: int, device: str):
        super().__init__()
        self.device = device
        self.n_decoder_layers: int = number_of_layers



class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, model_dimensions: int, number_of_heads: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        self.number_of_heads: int = number_of_heads
        self.model_dimensions: int = model_dimensions
        self.dropout_rate: float = dropout_rate

        if model_dimensions % number_of_heads != 0:
            raise ValueError("model_dimensions must be divisible by number of heads")
        self.head_dimension = model_dimensions // number_of_heads

        self.query_linear: Linear = nn.Linear(in_features=model_dimensions, out_features=model_dimensions)
        self.key_linear: Linear = nn.Linear(in_features=model_dimensions, out_features=model_dimensions)
        self.value_linear: Linear = nn.Linear(in_features=model_dimensions, out_features=model_dimensions)
        self.dropout: Dropout = nn.Dropout(p=self.dropout_rate)
        self.out_linear: Linear = nn.Linear(in_features=model_dimensions, out_features=model_dimensions)
        self.softmax: Softmax = nn.Softmax()

    def forward(self, value, key, query, do_dropout: bool = False, mask=None):

        queries: Tensor = self.query_linear(query)
        keys: Tensor = self.key_linear(key)
        values: Tensor = self.value_linear(value)

        queries: Tensor = queries.view(query.shape[0], query.shape[1], self.number_of_heads, self.head_dimension)
        keys: Tensor = keys.view(key.shape[0], key.shape[1], self.number_of_heads, self.head_dimension)
        values: Tensor = values.view(value.shape[0], value.shape[1], self.number_of_heads, self.head_dimension)

        queries = queries.transpose(1, 2).contiguous()
        keys = keys.transpose(1, 2).contiguous()
        values = values.transpose(1, 2).contiguous()

        scores: Tensor = torch.matmul(input=queries, other=keys.transpose(-2, -1)) / (self.model_dimensions ** (1/2))

        if mask is not None:
            mask = self.mask.unsqueeze()
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = self.softmax(scores, dim=-1)

        if do_dropout:
            scores = self.dropout(scores)

        out = torch.matmul(input=scores, other=values)
        out = out.transpose(1, 2).contiguous().view(query.size(0), -1, self.model_dimensions)
        out = self.out_linear(out)

        return out


class NormModule(nn.Module):
    def __init__(self, model_dimensions:int, eps=1e-6):
        super().__init__()

        self.size = model_dimensions
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm