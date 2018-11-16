import torch.nn as nn
import torch
from model.attention import LocationAttention
from model.module import ZoneoutLSTMEncoder, ZoneoutLSTMDecoder
from model.loss import sequence_mask
import pdb


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, num_chars, dim_embedding=512, dim_encoder=256, kernel_size=5, batch_size=32):
        super(Encoder, self).__init__()
        self.char_embedding = nn.Embedding(num_embeddings=num_chars, embedding_dim=dim_embedding, padding_idx=0)
        self.conv1 = ConvBlock(in_channels=dim_embedding, out_channels=dim_embedding, kernel_size=kernel_size)
        self.conv2 = ConvBlock(in_channels=dim_embedding, out_channels=dim_embedding, kernel_size=kernel_size)
        self.conv3 = ConvBlock(in_channels=dim_embedding, out_channels=dim_embedding, kernel_size=kernel_size)
        self.birnn = nn.LSTM(input_size=dim_embedding, hidden_size=dim_encoder, bidirectional=True, dropout=0.1)
        self.birnn_h0 = nn.Parameter(torch.randn(2, batch_size, dim_encoder), requires_grad=True)
        self.birnn_c0 = nn.Parameter(torch.randn(2, batch_size, dim_encoder), requires_grad=True)
        #self.birnn.flatten_parameters()
        #self.birnn = ZoneoutLSTMEncoder(input_size=dim_embedding, hidden_size=dim_encoder, zoneout_factor_cell=0.1,
        #                                zoneout_factor_hidden=0.1, bidirectional=True)

    def forward(self, text):
        # input - (batch, maxseqlen)
        x = self.char_embedding(text)  # (batch, seqlen, embdim)
        x = x.permute(0, 2, 1)  # swap to batch, channel, seqlen 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(2, 0, 1)  # swap seq, batch, dim for rnn
        x, _ = self.birnn(x, (self.birnn_h0, self.birnn_c0))    # 256 dims in either direction
        return x  # T x B x C


class PreNet(nn.Module):
    """
    Extracts 256d features from 80d input spectrogram frame
    """
    def __init__(self, in_features, out_features, dropout=0.5):
        super(PreNet, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(out_features, out_features)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, previous_y):
        """
        :param previous_y:  T x B x C
        :return:
        """
        x = self.relu1(self.fc1(previous_y))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        return x


class ConvTanhBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvTanhBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch = nn.BatchNorm1d(out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        :param x: T x B x C
        :return:
        """
        x = self.conv(x)
        x = self.batch(x)
        return self.tanh(x)


class PostNet(nn.Module):
    def __init__(self, num_mels, num_filters, kernel_size=5):
        super(PostNet, self).__init__()
        padding = kernel_size // 2
        out_channels = num_mels
        self.conv1 = ConvTanhBlock(in_channels=out_channels, out_channels=num_filters, kernel_size=kernel_size,
                                   padding=padding)
        self.conv2 = ConvTanhBlock(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size,
                                   padding=padding)
        self.conv3 = ConvTanhBlock(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size,
                                   padding=padding)
        self.conv4 = ConvTanhBlock(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size,
                                   padding=padding)
        self.conv5 = nn.Conv1d(in_channels=num_filters, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding)

    def forward(self, x):
        """
        :param x: [B x C x T]
        :return:
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x  # T x B x C


class Decoder(nn.Module):
    def __init__(self, max_decoder_steps, frames_per_step=1, dim_encoder=256, num_mels=80, dim_attention=128,
                 dim_decoder=1024, dim_prenet=256, num_layers=2, num_location_features=32, gate_threshold=0.5,
                 num_filters=512, kernel_size=5, batch_size=32):
        super(Decoder, self).__init__()
        self.num_mels = num_mels
        self.frames_per_step = frames_per_step
        self.dim_attention = dim_attention
        self.dim_decoder = dim_decoder
        self.dim_prenet = dim_prenet
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.num_layers = num_layers

        self.attention = LocationAttention(dim_key=dim_encoder, dim_query=dim_decoder, dim_attention=dim_attention,
                                           num_location_features=num_location_features)
        self.lstm = nn.LSTM(input_size=dim_prenet+dim_encoder, hidden_size=dim_decoder, num_layers=num_layers,
                dropout=0.1)
        #self.lstm = ZoneoutLSTMDecoder(input_size=dim_prenet+dim_encoder, hidden_size=dim_decoder,
        #                               zoneout_factor_cell=0.1, zoneout_factor_hidden=0.1, num_layers=num_layers)
        self.prenet = PreNet(in_features=num_mels, out_features=dim_prenet)
        self.mels_proj = nn.Linear(in_features=dim_decoder+dim_encoder, out_features=num_mels)
        self.stop_proj = nn.Linear(in_features=dim_decoder+dim_encoder, out_features=1)
        self.postnet = PostNet(num_mels=num_mels, num_filters=num_filters, kernel_size=kernel_size)
        self.decoder_output = nn.Parameter(torch.zeros(1, batch_size, num_mels))
        self.context = nn.Parameter(torch.zeros(1, batch_size, dim_encoder))
        self.decoder_hidden_state = nn.Parameter(torch.zeros(self.num_layers, batch_size, self.dim_decoder))
        self.decoder_cell_state = nn.Parameter(torch.zeros(self.num_layers, batch_size, self.dim_decoder))
        self.decoder_state = (self.decoder_hidden_state, self.decoder_cell_state)

    def init_decoder_output(self, batch_size):
        return torch.zeros([1, batch_size, self.num_mels]).cuda()


    def init_cum_alignment(self, batch_size, seq_len):
        return torch.zeros(batch_size, 1, seq_len).cuda()

    def init_context(self, step, batch_size, dim_memory):
        return torch.zeros(step, batch_size, dim_memory).cuda()

    def forward(self, encoder_output, txt_lengths, mels, is_training=True):
        """
        :param encoder_output:
        :param txt_lengths: also called memory length
        :param mels:
        :param is_training:
        :return:
        """
        if is_training:
            assert mels is not None
            batch_size, _, seq_len =  mels.size()
        else:
            _, batch_size, _ = encoder_output.size()
            assert batch_size == 1
            seq_len = self.max_decoder_steps

        seq_len_enc, _, dim_encoder = encoder_output.size()
        cum_alignment = self.init_cum_alignment(batch_size, seq_len_enc)
        mask_neg = ~sequence_mask(txt_lengths)
        mask_neg.requires_grad = False
        self.attention.init_state(encoder_output, mask_neg)

        mels_frame_list = []
        stop_token_list = []
        alignment_list = []
        decoder_frame_list = []
        context_tbc = self.context
        decoder_output = self.decoder_output
        decoder_state = self.decoder_state

        for t in range(seq_len):
            prev_output = self.prenet(decoder_output)                         # (1) x B x C(256)
            lstm_input = torch.cat([prev_output, context_tbc], dim=2)         # T(1) x B x C(768)
            lstm_out, decoder_state = self.lstm(lstm_input, decoder_state)    # T x B x C(1024)
            context_tbc, alignment, cum_alignment = self.attention(lstm_out, cum_alignment)
            project_input = torch.cat([lstm_out, context_tbc], dim=2)
            decoder_output = self.mels_proj(project_input)  # T(1) x B x C
            stop_token = self.stop_proj(project_input)   # T(1) x B x C(1)
            decoder_frame_list.append(decoder_output)
            stop_token_list.append(stop_token)
            alignment_list.append(alignment)

            if is_training:
                decoder_output = mels[:, :, t].unsqueeze(0)

            if not is_training and torch.sigmoid(stop_token[0, 0]) > self.gate_threshold:
                break

        decoder_frames = torch.stack(decoder_frame_list, dim=-1)  # 1 x B x C x T
        decoder_frames = decoder_frames.squeeze(0)  # B x C x T
        mels_frames = decoder_frames + self.postnet(decoder_frames) # B x C x T
        stop_tokens = torch.stack(stop_token_list, dim=-1)  # 1 x B x C x T
        stop_tokens = stop_tokens.squeeze(0)
        #pdb.set_trace()
        alignments = torch.stack(alignment_list, dim=-1)  # B x 1 x Te x Td
        alignments = alignments.squeeze(1)

        return mels_frames, decoder_frames, stop_tokens, alignments


class Tacotron2Net(nn.Module):

    def __init__(self, num_chars, max_decoder_steps, frames_per_step=1, dim_embedding=512, dim_encoder=256,
                 enc_kernel_size=5, num_mels=80, dim_attention=128, dim_decoder=1024, dim_prenet=256, num_layers=2,
                 num_location_features=32, gate_threshold=0.5, dec_num_filters=512, dec_kernel_size=5, batch_size=32):
        super(Tacotron2Net, self).__init__()
        self.encoder = Encoder(num_chars=num_chars, dim_embedding=dim_embedding, dim_encoder=dim_encoder,
                               kernel_size=enc_kernel_size, batch_size=batch_size)
        self.decoder = Decoder(max_decoder_steps=max_decoder_steps, frames_per_step=frames_per_step,
                               dim_encoder=dim_encoder*2, num_mels=num_mels, dim_attention=dim_attention,
                               dim_decoder=dim_decoder, dim_prenet=dim_prenet, num_layers=num_layers,
                               num_location_features=num_location_features, gate_threshold=gate_threshold,
                               num_filters=dec_num_filters, kernel_size=dec_kernel_size, batch_size=batch_size)

    def get_trainable_parameters(self):
        return self.parameters()

    def forward(self, text, txt_lengths, mels=None):
        is_training = False if mels is None else True
        encoder_output = self.encoder(text)
        frames, decoder_frames, stop_tokens, alignment = self.decoder(encoder_output, txt_lengths, mels=mels, is_training=is_training)
        return frames, decoder_frames, stop_tokens, alignment
