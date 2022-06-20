import torch
from torch import nn, Tensor

from models.modules import EmbeddingModule, PositionalEncoderModule, EncoderLayer, NormModule, DecoderLayer


class EncoderModule(nn.Module):
    """
    The whole Encoder
    Consists of Input embedding and multiple Encoder Layers
    """

    def __init__(self, vocabulary_size: int, model_dimensions: int, number_of_attention_heads: int, number_of_layers: int, max_seq_length: int, forward_expansion: int,
                 device: str):
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
    """
    The whole Decoder

    """

    def __init__(self, vocabulary_size: int, model_dimensions: int, number_of_attention_heads: int, number_of_layers: int, max_seq_length: int, forward_expansion: int,
                 device: str):
        super().__init__()
        self.device = device
        self.n_decoder_layers: int = number_of_layers

        self.embedder = EmbeddingModule(vocabulary_size=vocabulary_size, model_dimensions=model_dimensions)
        self.positional_encoder = PositionalEncoderModule(max_input_length=max_seq_length, model_dimensions=model_dimensions)

        decoder_blocks = []
        for i in range(number_of_layers):
            decoder_blocks.append(DecoderLayer(model_dimensions=model_dimensions, number_of_attention_heads=number_of_attention_heads, forward_expansion=forward_expansion))

        self.decoder_layers = nn.Sequential(*decoder_blocks)

        self.norm = NormModule(model_dimensions=model_dimensions)

    def forward(self, input, value_from_encoder, key_from_encoder, src_mask, target_mask):
        number_of_samples, seq_length = input.shape
        positions: Tensor = torch.arange(start=0, end=seq_length).expand(number_of_samples, seq_length).to(self.device)

        source_embedded = self.word_embedder(input)
        position_encodings = self.positional_encoder(positions)
        decoder_input = source_embedded + position_encodings

        x = self.decoder_layers(decoder_input, value_from_encoder, key_from_encoder, src_mask, target_mask)
        x = self.norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self,
                 src_vocabulary,
                 target_vocabulary,
                 model_dimensions: int,
                 number_of_attention_heads: int = 8,
                 number_of_layers: int = 6,
                 forward_expansion: int = 4,
                 max_seq_length: int = 100,
                 device: str = "cpu"):
        super().__init__()
        self.encoder = EncoderModule(
            vocabulary_size=src_vocabulary,
            model_dimensions=model_dimensions,
            number_of_attention_heads=number_of_attention_heads,
            number_of_layers=number_of_layers,
            max_seq_length=max_seq_length,
            forward_expansion=forward_expansion
        )

        self.decoder = DecoderModule(
            vocabulary_size=target_vocabulary,
            model_dimensions=model_dimensions,
            number_of_attention_heads=number_of_attention_heads,
            number_of_layers=number_of_layers,
            max_seq_length=max_seq_length,
            forward_expansion=forward_expansion,
            device=device)

        self.out = nn.Linear(model_dimensions, target_vocabulary)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

