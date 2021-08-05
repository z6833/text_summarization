import tensorflow as tf

from src.build_seq2seq_transformer.encoders.self_attention_encoder import EncoderLayer
from src.build_seq2seq_transformer.decoders.self_attention_decoder import DecoderLayer
from src.build_seq2seq_transformer.layers.transformer import Embedding, create_masks


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(input_vocab_size, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        x = self.embedding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        # (batch_size, input_seq_len, d_model)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.depth = self.d_model // self.num_heads
        self.embedding = Embedding(target_vocab_size, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        attention_weights = {}

        x = self.embedding(x)
        out = self.dropout(x, training=training)

        for i in range(self.num_layers):
            out, block1, block2 = self.dec_layers[i](out, enc_output, training,
                                                     look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        p_gens = None
        return out, attention_weights, p_gens


class PGN_TRANSFORMER(tf.keras.Model):
    def __init__(self, params):
        super(PGN_TRANSFORMER, self).__init__()

        self.num_blocks = params["num_blocks"]
        self.batch_size = params["batch_size"]
        self.vocab_size = params["vocab_size"]
        self.num_heads = params["num_heads"]

        self.encoder = Encoder(params["num_blocks"],
                               params["d_model"],
                               params["num_heads"],
                               params["dff"],
                               params["vocab_size"],
                               params["dropout_rate"])

        self.decoder = Decoder(params["num_blocks"],
                               params["d_model"],
                               params["num_heads"],
                               params["dff"],
                               params["vocab_size"],
                               params["dropout_rate"])

        self.final_layer = tf.keras.layers.Dense(params["vocab_size"])

    def call(self, inp, extended_inp, max_oov_len, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights, p_gens = self.decoder(tar,
                                                             enc_output,
                                                             training,
                                                             look_ahead_mask,
                                                             dec_padding_mask)
        final_output = self.final_layer(dec_output)
        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = tf.nn.softmax(final_output)

        attn_dists = attention_weights['decoder_layer{}_block2'.format(self.num_blocks)]

        # (batch_size,num_heads, targ_seq_len, inp_seq_len)
        attn_dists = tf.reduce_sum(attn_dists, axis=1) / self.num_heads

        # outputs = dict(logits=tf.stack(final_dists, 1), attentions=attn_dists)
        outputs = dict(logits=final_output, attentions=attn_dists)
        return outputs


if __name__ == '__main__':
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, input_vocab_size=8500)

    embedding = Embedding(vocab_size=30000,
                          d_model=512)
    x = tf.random.uniform((64, 62), maxval=10, dtype=tf.int32)
    enc_inp = embedding(x)

    sample_encoder_output = sample_encoder(enc_inp,
                                           training=False, mask=None)

    # (batch_size, input_seq_len, d_model)
    print(sample_encoder_output.shape)

    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, target_vocab_size=8000)

    y = tf.random.uniform((64, 26), maxval=10, dtype=tf.int32)
    dec_inp = embedding(y)

    output, attn, p_gens = sample_decoder(dec_inp,
                                          enc_output=sample_encoder_output,
                                          training=False, look_ahead_mask=None,
                                          padding_mask=None)

    print(output.shape, attn['decoder_layer2_block2'].shape)

    params = {}
    params["num_blocks"] = 2
    params["d_model"] = 512
    params["num_heads"] = 8
    params["dff"] = 32
    params["vocab_size"] = 30000
    params["dropout_rate"] = 0.1
    params["num_blocks"] = 2
    params["batch_size"] = 64

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(x, y)

    print('enc_padding_mask:{},combined_mask:{},dec_padding_mask:{}'.format(enc_padding_mask.shape, combined_mask.shape,
                                                                            dec_padding_mask.shape))

    transformer = PGN_TRANSFORMER(params)

    outputs = transformer(x,
                          extended_inp=x,
                          max_oov_len=0,
                          tar=y,
                          training=True,
                          enc_padding_mask=enc_padding_mask,
                          look_ahead_mask=combined_mask,
                          dec_padding_mask=dec_padding_mask)

    print(outputs)
