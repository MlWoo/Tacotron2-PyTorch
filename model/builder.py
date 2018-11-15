# coding: utf-8
from __future__ import with_statement, print_function, absolute_import


def Tacotron2(num_chars=32,
              max_decoder_steps=800,
              frames_per_step=1,
              dim_embedding=512,
              dim_encoder=256,
              enc_kernel_size=5,
              num_mels=80,
              dim_attention=128,
              dim_decoder=1024,
              dim_prenet=256,
              num_layers=2,
              num_location_features=32,
              gate_threshold=0.5,
              dec_num_filters=512,
              dec_kernel_size=5
              ):
    from model.tacotron2 import Tacotron2Net

    model = Tacotron2Net(num_chars=num_chars, max_decoder_steps=max_decoder_steps, frames_per_step=frames_per_step,
                         dim_embedding=dim_embedding, dim_encoder=dim_encoder, enc_kernel_size=enc_kernel_size,
                         num_mels=num_mels, dim_attention=dim_attention, dim_decoder=dim_decoder, dim_prenet=dim_prenet,
                         num_layers=num_layers, num_location_features=num_location_features,
                         gate_threshold=gate_threshold, dec_num_filters=dec_num_filters,
                         dec_kernel_size=dec_kernel_size)

    return model
