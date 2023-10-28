import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d, Conv3d
from .transformer_encoder import TransformerEncoder


class Wav2Lip(nn.Module):
    def __init__(self, d_model=512):
        super(Wav2Lip, self).__init__()
        self.d_model = d_model

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv3d(6, 16, kernel_size=7, stride=1, padding=3)),  # 96,96

            nn.Sequential(Conv3d(16, 32, kernel_size=3, stride=(2, 2, 1), padding=1),  # 48,48
                          Conv3d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv3d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv3d(32, 64, kernel_size=3, stride=(2, 2, 1), padding=1),  # 24,24
                          Conv3d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv3d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv3d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv3d(64, 128, kernel_size=3, stride=(2, 2, 1), padding=1),  # 12,12
                          Conv3d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv3d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv3d(128, 256, kernel_size=3, stride=(2, 2, 1), padding=1),  # 6,6
                          Conv3d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv3d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv3d(256, 512, kernel_size=3, stride=(2, 2, 1), padding=1),  # 3,3
                          Conv3d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),
            nn.Sequential(Conv3d(512, 512, kernel_size=3, stride=1, padding=(0, 0, 1)),  # 1, 1
                          Conv3d(512, 512, kernel_size=1, stride=1, padding=0)),])
            # Conv3d(6, layers[0], kernel_size=7, stride=1, padding=3),
            # # 64.48.48.5
            # Conv3d(layers[0], layers[1], kernel_size=3, stride=(2, 2, 1), padding=1),
            # Conv3d(layers[1], layers[1], kernel_size=3, stride=1, padding=1, residual=True),
            # Conv3d(layers[1], layers[1], kernel_size=3, stride=1, padding=1, residual=True),
            # # 128.24.24.5
            # Conv3d(layers[1], layers[2], kernel_size=3, stride=(2, 2, 1), padding=1),
            # Conv3d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),
            # Conv3d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),
            # Conv3d(layers[2], layers[2], kernel_size=3, stride=1, padding=1, residual=True),
            # # 256.12.12.5
            # Conv3d(layers[2], layers[3], kernel_size=3, stride=(2, 2, 1), padding=1),
            # Conv3d(layers[3], layers[3], kernel_size=3, stride=1, padding=1, residual=True),
            # Conv3d(layers[3], layers[3], kernel_size=3, stride=1, padding=1, residual=True),
            # # 512.6.6.5
            # Conv3d(layers[3], layers[4], kernel_size=3, stride=(2, 2, 1), padding=1),
            # Conv3d(layers[4], layers[4], kernel_size=3, stride=1, padding=1, residual=True),
            # Conv3d(layers[4], layers[4], kernel_size=3, stride=1, padding=1, residual=True),
            # # 512.3.3.5---512.1.1.5----512.1.1.5
            # Conv3d(layers[4], layers[4], kernel_size=3, stride=(2, 2, 1), padding=1),
            # Conv3d(layers[4], layers[4], kernel_size=3, stride=1, padding=(0, 0, 1)),
            # Conv3d(layers[4], layers[4], kernel_size=1, stride=1, padding=0),
        #
        # )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
# 9.6
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

        # self.av_transformer = TransformerEncoder(embed_dim=d_model,
        #                                          num_heads=8,
        #                                          layers=4,
        #                                          attn_dropout=0.0,
        #                                          relu_dropout=0.1,
        #                                          res_dropout=0.1,
        #                                          embed_dropout=0.25,
        #                                          attn_mask=True)
        # self.va_transformer = TransformerEncoder(embed_dim=d_model,
        #                                          num_heads=8,
        #                                          layers=4,
        #                                          attn_dropout=0.0,
        #                                          relu_dropout=0.1,
        #                                          res_dropout=0.1,
        #                                          embed_dropout=0.25,
        #                                          attn_mask=True)
        # self.trans = nn.TransformerEncoder()

        self.mem_transformer = TransformerEncoder(embed_dim=d_model,
                                                  num_heads=8,
                                                  layers=4,
                                                  attn_dropout=0.0,
                                                  relu_dropout=0.1,
                                                  res_dropout=0.1,
                                                  embed_dropout=0.25,
                                                  attn_mask=True)

        # self.face_decoder_blocks = nn.ModuleList([
            # nn.Sequential(Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0), ),
            #
            # nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  # 3,3
            #               Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),
            #
            # nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            #               Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            #               Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            #               ),  # 6, 6
            #
            # # nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            # #               Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            # #               Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), ),  # 12, 12
            #
            # nn.Sequential(Conv2dTranspose(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            #               Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            #               Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), ),  # 12.12
            #
            # nn.Sequential(Conv2dTranspose(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            #               Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            #               Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),  # 24, 24
            #
            # nn.Sequential(Conv2dTranspose(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            #               Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            #               Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), ),
            #
            # nn.Sequential(Conv2dTranspose(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            #               Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True),
            #               Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True), ),#96.96
            # ])
        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0), ),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  # 6, 6

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), ),  # 12, 12

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), ),  # 24, 24

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), ),  # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), ), ])  # 96,96

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            # face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        face_sequences = face_sequences.permute(0, 1, 3, 4, 2).contiguous()
        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1

        face_embedding = self.face_encoder_blocks(face_sequences)

        vid_embedding = face_embedding.squeeze(2).squeeze(2)

        aud_embedding = audio_embedding.squeeze(2)
        aud_embedding = aud_embedding.view(B, 512, -1).contiguous()
        vid_embedding = vid_embedding.permute(2, 0, 1).contiguous()
        aud_embedding = aud_embedding.permute(2, 0, 1).contiguous()

        av_embedding = self.av_transformer(aud_embedding, vid_embedding, vid_embedding)
        va_embedding = self.va_transformer(vid_embedding, aud_embedding, aud_embedding)

        tranformer_out = self.mem_transformer(av_embedding, va_embedding, va_embedding)
        tranformer_out = torch.cat([tranformer_out[:, i] for i in range(tranformer_out.size(1))], dim=0)
        tranformer_out = tranformer_out.unsqueeze(2).unsqueeze(2)

        x = torch.cat((audio_embedding, tranformer_out),dim=1)
        for f in self.face_decoder_blocks:
            x = f(x)
        x = self.output_block(x)
        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs
5.8.512
        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            y = torch.cat([x[:, :, :, :, i] for i in range(x.size(4))], dim=0)
            feats.append(y)

        x = audio_embedding
        x = x.squeeze(2).permute(0, 2, 1)#40.1.512
        x = self.mem_transformer(x)
        x = x.permute(0, 2, 1).unsqueeze(2)


        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs


class Wav2Lip_disc_qual(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)),  # 48,96

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),  # 48,48
                          nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 24,24
                          nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # 12,12
                          nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 6,6
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                          nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1), ),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                          nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences
# 感知损失--判别器
    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1),
                                                 torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)

