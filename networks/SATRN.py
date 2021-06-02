from copy import deepcopy
import math
import random
import operator
from queue import PriorityQueue
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys # for DEBUG
sys.path.insert(0, '/opt/ml/code') # for DEBUG
from dataset import START, PAD
from beam_search import BeamSearchNode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BottleneckBlock(nn.Module):
    """
    Dense Bottleneck Block

    It contains two convolutional layers, a 1x1 and a 3x3.
    """

    def __init__(self, input_size, growth_rate, dropout_rate=0.2, num_bn=3):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added. That is the ouput
                size of the last convolutional layer.
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(BottleneckBlock, self).__init__()
        inter_size = num_bn * growth_rate
        self.norm1 = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            input_size, inter_size, kernel_size=1, stride=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(inter_size)
        self.conv2 = nn.Conv2d(
            inter_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    """
    Transition Block

    A transition layer reduces the number of feature maps in-between two bottleneck
    blocks.
    """

    def __init__(self, input_size, output_size):
        """
        Args:
            input_size (int): Number of channels of the input
            output_size (int): Number of channels of the output
        """
        super(TransitionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            input_size, output_size, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        return self.pool(out)


class DenseBlock(nn.Module):
    """
    Dense block

    A dense block stacks several bottleneck blocks.
    """

    def __init__(self, input_size, growth_rate, depth, dropout_rate=0.2):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added per bottleneck block
            depth (int): Number of bottleneck blocks
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(DenseBlock, self).__init__()
        layers = [
            BottleneckBlock(
                input_size + i * growth_rate, growth_rate, dropout_rate=dropout_rate
            )
            for i in range(depth)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DeepCNN300(nn.Module):
    """
    This is specialized to the math formula recognition task
    - three convolutional layers to reduce the visual feature map size and to capture low-level visual features
    (128, 256) -> (8, 32) -> total 256 features
    - transformer layers cannot change the channel size, so it requires a wide feature dimension
    ***** this might be a point to be improved !!
    """

    def __init__(
        self, input_channel, num_in_features, output_channel=256, dropout_rate=0.2, depth=16, growth_rate=24
    ):
        super(DeepCNN300, self).__init__()
        self.conv0 = nn.Conv2d(
            input_channel,  # 3
            num_in_features,  # 48
            kernel_size=7,
            stride=2,
            padding=3, # 7//2
            bias=False,
        )
        self.norm0 = nn.BatchNorm2d(num_in_features)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # 1/4 (128, 128) -> (32, 32)
        num_features = num_in_features

        self.block1 = DenseBlock(
            num_features,  # 48
            growth_rate=growth_rate,  # 48 + growth_rate(24)*depth(16) -> 432
            depth=depth,  # 16?
            dropout_rate=0.2,
        )
        num_features = num_features + depth * growth_rate
        self.trans1 = TransitionBlock(num_features, num_features // 2)  # 16 x 16
        num_features = num_features // 2
        self.block2 = DenseBlock(
            num_features,  # 128
            growth_rate=growth_rate,  # 16
            depth=depth,  # 8
            dropout_rate=0.2,
        )
        num_features = num_features + depth * growth_rate
        self.trans2_norm = nn.BatchNorm2d(num_features)
        self.trans2_relu = nn.ReLU(inplace=True)
        self.trans2_conv = nn.Conv2d(
            num_features, num_features // 2, kernel_size=1, stride=1, bias=False  # 128
        )

    def forward(self, input):
        out = self.conv0(input)  # (H, V, )
        out = self.relu(self.norm0(out))
        out = self.max_pool(out)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out_before_trans2 = self.trans2_relu(self.trans2_norm(out))
        out_A = self.trans2_conv(out_before_trans2)
        return out_A  # 128 x (16x16)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask=mask, value=float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, q_channels, k_channels, head_num=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.q_channels = q_channels
        self.k_channels = k_channels
        self.head_dim = q_channels // head_num
        self.head_num = head_num

        self.q_linear = nn.Linear(q_channels, self.head_num * self.head_dim)
        self.k_linear = nn.Linear(k_channels, self.head_num * self.head_dim)
        self.v_linear = nn.Linear(k_channels, self.head_num * self.head_dim)
        self.attention = ScaledDotProductAttention(
            temperature=(self.head_num * self.head_dim) ** 0.5, dropout=dropout
        ) # [?] (d_k)^{0.5}가 되어야하는 것 아닌가?
        self.out_linear = nn.Linear(self.head_num * self.head_dim, q_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        b, q_len, k_len, v_len = q.size(0), q.size(1), k.size(1), v.size(1)
        q = (
            self.q_linear(q)
            .view(b, q_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(k)
            .view(b, k_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(v)
            .view(b, v_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )

        if mask is not None:
            mask = mask.unsqueeze(1)

        out, attn = self.attention(q, k, v, mask=mask)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(b, q_len, self.head_num * self.head_dim)
        )
        out = self.out_linear(out)
        out = self.dropout(out)

        return out


class Feedforward(nn.Module):
    def __init__(self, filter_size=2048, hidden_dim=512, dropout=0.1):
        super(Feedforward, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, filter_size, True),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(filter_size, hidden_dim, True),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        ) # 일부러 차원을 높였다가 낮추는건가?

    def forward(self, input):
        return self.layers(input)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_size, filter_size, head_num, dropout_rate=0.2):
        super(TransformerEncoderLayer, self).__init__()

        self.attention_layer = MultiHeadAttention(
            q_channels=input_size,
            k_channels=input_size,
            head_num=head_num,
            dropout=dropout_rate,
        )
        self.attention_norm = nn.LayerNorm(normalized_shape=input_size)
        self.feedforward_layer = Feedforward(
            filter_size=filter_size, hidden_dim=input_size
        )
        self.feedforward_norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, input):

        att = self.attention_layer(input, input, input)
        out = self.attention_norm(att + input)

        ff = self.feedforward_layer(out)
        out = self.feedforward_norm(ff + out)
        return out


class PositionalEncoding2D(nn.Module):
    def __init__(self, in_channels, max_h=64, max_w=128, dropout=0.1):
        super(PositionalEncoding2D, self).__init__()

        self.h_position_encoder = self.generate_encoder(in_channels // 2, max_h)
        self.w_position_encoder = self.generate_encoder(in_channels // 2, max_w)

        self.h_linear = nn.Linear(in_channels // 2, in_channels // 2)
        self.w_linear = nn.Linear(in_channels // 2, in_channels // 2)

        self.dropout = nn.Dropout(p=dropout)

    def generate_encoder(self, in_channels, max_len):
        pos = torch.arange(max_len).float().unsqueeze(1)
        i = torch.arange(in_channels).float().unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_channels)
        position_encoder = pos * angle_rates
        position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
        position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])
        return position_encoder  # (Max_len, In_channel)

    def forward(self, input):
        ### Require DEBUG <- 개극혐;
        b, c, h, w = input.size()
        h_pos_encoding = (
            self.h_position_encoder[:h, :].unsqueeze(1).to(input.get_device())
        )
        # h_pos_encoding = (
        #     self.h_position_encoder[:h, :].unsqueeze(1).to(device)
        # )
        h_pos_encoding = self.h_linear(h_pos_encoding)  # [H, 1, D]

        w_pos_encoding = (
            self.w_position_encoder[:w, :].unsqueeze(0).to(input.get_device())
        )
        # w_pos_encoding = (
        #     self.w_position_encoder[:w, :].unsqueeze(0).to(device)
        # )
        w_pos_encoding = self.w_linear(w_pos_encoding)  # [1, W, D]

        h_pos_encoding = h_pos_encoding.expand(-1, w, -1)   # h, w, c/2
        w_pos_encoding = w_pos_encoding.expand(h, -1, -1)   # h, w, c/2

        pos_encoding = torch.cat([h_pos_encoding, w_pos_encoding], dim=2)  # [H, W, 2*D]
        pos_encoding = pos_encoding.permute(2, 0, 1)  # [2*D, H, W]

        out = input + pos_encoding.unsqueeze(0)
        out = self.dropout(out)

        return out


class TransformerEncoderFor2DFeatures(nn.Module):
    """
    Transformer Encoder for Image
    1) ShallowCNN : low-level visual feature identification and dimension reduction
    2) Positional Encoding : adding positional information to the visual features
    3) Transformer Encoders : self-attention layers for the 2D feature maps
    """

    def __init__(
        self,
        input_size,
        hidden_dim,
        filter_size,
        head_num,
        layer_num,
        dropout_rate=0.1,
        checkpoint=None,
    ):
        super(TransformerEncoderFor2DFeatures, self).__init__()

        self.shallow_cnn = DeepCNN300(
            input_size,
            num_in_features=48,
            output_channel=hidden_dim,
            dropout_rate=dropout_rate,
        )
        self.positional_encoding = PositionalEncoding2D(hidden_dim)
        self.attention_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_dim, filter_size, head_num, dropout_rate)
                for _ in range(layer_num)
            ]
        )
        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def forward(self, input):

        out = self.shallow_cnn(input)  # [b, c, h, w]
        out = self.positional_encoding(out)  # [b, c, h, w]

        # flatten
        b, c, h, w = out.size()
        out = out.view(b, c, h * w).transpose(1, 2)  # [b, h x w, c]

        for layer in self.attention_layers:
            out = layer(out)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_size, src_size, filter_size, head_num, dropout_rate=0.2):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attention_layer = MultiHeadAttention(
            q_channels=input_size,
            k_channels=input_size,
            head_num=head_num,
            dropout=dropout_rate,
        )
        self.self_attention_norm = nn.LayerNorm(normalized_shape=input_size)

        self.attention_layer = MultiHeadAttention(
            q_channels=input_size,
            k_channels=src_size,
            head_num=head_num,
            dropout=dropout_rate,
        )
        self.attention_norm = nn.LayerNorm(normalized_shape=input_size)

        self.feedforward_layer = Feedforward(
            filter_size=filter_size, hidden_dim=input_size
        )
        self.feedforward_norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, tgt, tgt_prev, src, tgt_mask):
        # Train
        if tgt_prev == None:  
            att = self.self_attention_layer(q=tgt, k=tgt, v=tgt, mask=tgt_mask)
            out = self.self_attention_norm(att+tgt) # element-wise addition

            att = self.attention_layer(q=tgt, k=src, v=src)
            out = self.attention_norm(att+out)

            ff = self.feedforward_layer(out)
            out = self.feedforward_norm(ff+out)
        else:
            tgt_prev = torch.cat([tgt_prev, tgt], 1)
            att = self.self_attention_layer(q=tgt, k=tgt_prev, v=tgt_prev, mask=tgt_mask)
            out = self.self_attention_norm(att + tgt)

            att = self.attention_layer(tgt, src, src)
            out = self.attention_norm(att + out)

            ff = self.feedforward_layer(out)
            out = self.feedforward_norm(ff + out)
        return out


class PositionEncoder1D(nn.Module):
    def __init__(self, in_channels, max_len=500, dropout=0.1):
        super(PositionEncoder1D, self).__init__()

        self.position_encoder = self.generate_encoder(in_channels, max_len) # [MAX_LEN, IN_CHANNELS(HIDDEN)]
        self.position_encoder = self.position_encoder.unsqueeze(0) # [1, MAX_LEN, IN_CHANNELS(HIDDEN)]
        self.dropout = nn.Dropout(p=dropout)

    def generate_encoder(self, in_channels, max_len):
        pos = torch.arange(max_len).float().unsqueeze(1) # [MAX_LEN, 1]
        i = torch.arange(in_channels).float().unsqueeze(0) # [1, IN_CHANNELS]
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_channels) # [1, IN_CHANNELS]

        position_encoder = pos * angle_rates # [MAX_LEN, IN_CHANNELS], broad-casting
        position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2]) # {2i}th - SIN
        position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2]) # {2i+1}th - COS

        return position_encoder

    def forward(self, x, point=-1) -> torch.Tensor: # [?]
        """input 텐서(x)에 PE 벡터를 적용하는 함수. Sinusoidal PE를 통해 x의 길이에 관계 없이 unique한 위치 정보를 전달

        Args:
            x (torch.Tensor): P.E를 적용할 텐서, [B, 1, HIDDEN]
            point (int, optional):
                - -1 입력: 인풋 텐서 길이에 맞게 슬라이싱하여 PE 적용
                - 특정 인덱스 입력: 특정 PE 벡터를 가져와 인풋 텐서에 적용
                - Defaults to -1.

        Returns:
            [torch.Tensor]: PE가 적용된 인풋 텐서
        """
        if point == -1:
            out = x + self.position_encoder[:, : x.size(1), :].to(x.get_device())
            # out = x + self.position_encoder[:, : x.size(1), :].to(device) # 
            out = self.dropout(out)
        else: # 특정 위치에 대한 P.E를 더함
            out = x + self.position_encoder[:, point, :].unsqueeze(1).to(x.get_device()) # [1, 1, IN_CHANNELS]
            # out = x + self.position_encoder[:, point, :].unsqueeze(1).to(device) # [1, 1, IN_CHANNELS]
        return out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_classes,
        src_dim,
        hidden_dim,
        filter_dim,
        head_num,
        dropout_rate,
        pad_id,
        st_id,
        layer_num=1,
        checkpoint=None,
    ):
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(num_classes + 1, hidden_dim)
        self.hidden_dim = hidden_dim
        self.filter_dim = filter_dim
        self.num_classes = num_classes
        self.layer_num = layer_num

        self.pos_encoder = PositionEncoder1D(
            in_channels=hidden_dim, dropout=dropout_rate
        )

        self.attention_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    input_size=hidden_dim,
                    src_size=src_dim,
                    filter_size=filter_dim,
                    head_num=head_num,
                    dropout_rate=dropout_rate
                )
                for _ in range(layer_num)
            ]
        )
        self.generator = nn.Linear(hidden_dim, num_classes)

        self.pad_id = pad_id
        self.st_id = st_id

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def pad_mask(self, text):
        pad_mask = text == self.pad_id
        pad_mask[:, 0] = False
        pad_mask = pad_mask.unsqueeze(1)

        return pad_mask

    def order_mask(self, length):
        """대각성분을 기준으로 위는 True, 아래는 False인 Masking Square Matrix 생성

        Args:
            length ([type]): Matrix 크기

        Returns:
            torch.Tensor: Masking Square Matrix, [1, LENGTH, LENGTH]
        """
        order_mask = torch.triu(torch.ones(length, length), diagonal=1).bool()
        order_mask = order_mask.unsqueeze(0).to(device)
        return order_mask

    def text_embedding(self, texts):
        """R^{VOCAB_SIZE} -> R^{HIDDEN_SIZE}"""
        tgt = self.embedding(texts)
        tgt *= math.sqrt(tgt.size(2))
            # PE 적용 전 임베딩 값을 증폭시키기 위해 상수배
            # 임베딩 값의 의미 손실 방지를 위해

        return tgt

    def forward(
        self, src, text, is_train=True, batch_max_length=50, teacher_forcing_ratio=1.0
    ):
        # teacher forcing
        if is_train and random.random() < teacher_forcing_ratio:
            tgt = self.text_embedding(text)
            tgt = self.pos_encoder(tgt)
            tgt_mask = self.pad_mask(text) | self.order_mask(text.size(1))
            for layer in self.attention_layers:
                tgt = layer(tgt, None, src, tgt_mask)
            out = self.generator(tgt)

        # no teacher forcing
        else:
            out = []
            num_steps = batch_max_length - 1
            target = torch.LongTensor(src.size(0)).fill_(self.st_id).to(device) # [START] token
            features = [None] * self.layer_num # 

            for t in range(num_steps):
                target = target.unsqueeze(1) # [B, 1], 한 글자 단위
                tgt = self.text_embedding(texts=target) # [B, 1, HIDDEN]
                tgt = self.pos_encoder(x=tgt, point=t) # [B, 1, HIDDEN]
                tgt_mask = self.order_mask(length=t+1) # [1, LEN, LEN]
                tgt_mask = tgt_mask[:, -1].unsqueeze(1)  # [1, 1, LEN] <- 없어도 될 것 같은데

                for l, layer in enumerate(self.attention_layers):
                    tgt = layer(tgt=tgt, tgt_prev=features[l], src=src, tgt_mask=tgt_mask) # [B, 1, HIDDEN]
                    features[l] = (
                        tgt if features[l] == None else torch.cat([features[l], tgt], 1)
                    )
                    # 첫 state: <SOS> state를 넣음, [B, 1, HIDDEN]
                    # 이후 state: 이전 state를 하나씩 쌓아감(torch.cat)
                        # [B, 1, HIDDEN] -> [B, 2, HIDDEN] -> [B, 3, HIDDEN] -> ...

                _out = self.generator(tgt)  # [B, 1, VOCAB_SIZE]
                target = torch.argmax(_out[:, -1:, :], dim=-1)  # [B, 1], 샘플별 최대 확률 토큰ID
                target = target.squeeze()   # [B]
                out.append(_out)
            
            out = torch.stack(out, dim=1).to(device)    # [b, max length, 1, class length]
            out = out.squeeze(2)    # [b, max length, class length]

        return out


class SATRN(nn.Module):
    def __init__(self, FLAGS, train_dataset, checkpoint=None):
        super(SATRN, self).__init__()

        self.encoder = TransformerEncoderFor2DFeatures(
            input_size=FLAGS.data.rgb,
            hidden_dim=FLAGS.SATRN.encoder.hidden_dim,
            filter_size=FLAGS.SATRN.encoder.filter_dim,
            head_num=FLAGS.SATRN.encoder.head_num,
            layer_num=FLAGS.SATRN.encoder.layer_num,
            dropout_rate=FLAGS.dropout_rate,
        )

        self.decoder = TransformerDecoder(
            num_classes=len(train_dataset.id_to_token),
            src_dim=FLAGS.SATRN.decoder.src_dim,
            hidden_dim=FLAGS.SATRN.decoder.hidden_dim,
            filter_dim=FLAGS.SATRN.decoder.filter_dim,
            head_num=FLAGS.SATRN.decoder.head_num,
            dropout_rate=FLAGS.dropout_rate,
            pad_id=train_dataset.token_to_id[PAD],
            st_id=train_dataset.token_to_id[START],
            layer_num=FLAGS.SATRN.decoder.layer_num,
        )

        self.criterion = (
            nn.CrossEntropyLoss()
        )  # without ignore_index=train_dataset.token_to_id[PAD]

        if checkpoint:
            self.load_state_dict(checkpoint)

    def forward(self, input, expected, is_train, teacher_forcing_ratio):
        enc_result = self.encoder(input)
        dec_result = self.decoder(
            src=enc_result,
            text=expected[:, :-1],
            is_train=is_train,
            batch_max_length=expected.size(1),
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return dec_result

    def beam_search(
        self, 
        input: torch.Tensor, 
        data_loader: DataLoader,
        topk: int=1, 
        beam_width: int=5, 
        max_sequence: int=230
        ):
        # 사용할 토큰
        sos_token_id = train_data_loader.dataset.token_to_id['<SOS>']
        eos_token_id = train_data_loader.dataset.token_to_id['<EOS>']
        pad_token_id = train_data_loader.dataset.token_to_id['<PAD>']

        batch_size = len(input)
        src = self.encoder(input) # [B, HxW, C]

        decoded_batch = []
        with torch.no_grad():

            # 문장 단위 생성
            for data_idx in range(batch_size):

                end_nodes = []
                number_required = min((topk + 1), topk - len(end_nodes))  # 최대 생성 횟수

                # 빔서치 과정 상 역추적을 위한 우선순위큐 선언
                nodes = PriorityQueue()

                # 시작 토큰 초기화
                current_src = src[data_idx, :, :].unsqueeze(0) # [B=1, HxW, C]
                current_input = torch.LongTensor([sos_token_id]) # [B=1]
                current_hidden = [None] * self.decoder.layer_num
                node = BeamSearchNode(
                    hidden_state=deepcopy(current_hidden),
                    prev_node=None,
                    token_id=deepcopy(current_input), # [1]
                    log_prob=0,
                    length=1 # NOTE: P.E에 사용
                )
                score = -node.eval()

                # 최대힙: 확률 높은 토큰을 추출하기 위함
                nodes.put((score, node)) 

                num_steps = 0
                while True:
                    if num_steps >= (max_sequence-1)*beam_width:
                        break

                    # 최대확률샘플 추출/제거, score: 로그확률, n: BeamSearchNode
                    score, n = nodes.get()
                    current_input = n.token_id # [B=1]
                    current_hidden = n.hidden_state
                    current_point = n.len - 1 # P.E 적용 시 활용
                    assert current_input.ndim == 1
                    assert len(current_hidden) == self.decoder.layer_num

                    # 종료 토큰이 생성될 경우(종료 토큰 & 이전 노드 존재)
                    if n.token_id.item() == eos_token_id and n.prev_node != None:
                        end_nodes.append((score, n))
                        if len(end_nodes) >= number_required:
                            break
                        else:
                            continue
                        
                    current_input = current_input.unsqueeze(1) # [B=1, 1]
                    assert current_input.ndim == 2
                    
                    tgt = self.decoder.text_embedding(texts=current_input.to(input.get_device())) # [B=1, 1, HIDDEN]
                    assert tgt.size(-1) == self.decoder.hidden_dim # [B=1, 1, HIDDEN]
                    assert tgt.ndim == 3 # TODO. DEBUG
                    
                    tgt = self.decoder.pos_encoder(x=tgt, point=current_point) # [B=1, 1, HIDDEN]
                    assert tgt.size(-1) == self.decoder.hidden_dim
                    assert tgt.ndim == 3

                    tgt_mask = self.decoder.order_mask(length=current_point+1) # [B=1, LEN, LEN]
                    assert tgt_mask.ndim == 3
                    assert tgt_mask.size(-1) == current_point+1
                    assert tgt_mask.size(-2) == current_point+1

                    tgt_mask = tgt_mask[:, -1].unsqueeze(1) # [B=1, 1, LEN]
                    assert tgt_mask.ndim == 3
                    assert tgt_mask.size(-1) == current_point+1
                    assert tgt_mask.size(-2) == 1

                    # 어텐션 레이어 통과
                    for l, layer in enumerate(self.decoder.attention_layers):
                        tgt = layer(
                            tgt=tgt, # [B=1, 1, HIDDEN]
                            tgt_prev=current_hidden[l], 
                            src=current_src, 
                            tgt_mask=tgt_mask
                            ) # [1, 1, HIDDEN]
                        assert tgt.ndim == 3
                        assert tgt.size(-1) == self.decoder.hidden_dim

                        # Hidden state 갱신
                        # 첫 state: [1, 1, HIDDEN]
                        # 이후: [B=1, 1, HIDDEN] -> [B=1, 2, HIDDEN] -> [B=1, 3, HIDDEN] -> ...
                        current_hidden[l] = (tgt if current_hidden[l] is None else torch.cat([current_hidden[l], tgt], dim=1))
                        assert current_hidden[l].ndim == 3

                    # 확률화하기 전 모델의 로짓
                    prob_step = self.decoder.generator(tgt) # [B=1, 1, VOCAB_SIZE]
                    assert prob_step.ndim == 3
                    assert prob_step.size(-1) == self.decoder.num_classes

                    # 모델의 로짓을  확률화
                    log_prob_step = F.log_softmax(prob_step, dim=-1) # [B=1, 1, VOCAB_SIZE]
                    log_prob, indices = torch.topk(log_prob_step, beam_width)
                    assert indices.ndim == 3
                    assert indices.size(-1) == beam_width
                    assert log_prob.ndim == 3
                    assert log_prob.size(-1) == beam_width

                    # 다음 state에 활용할 {beam_width}개 후보 노드를 우선순위큐에 삽입
                    next_nodes = []
                    for new_k in range(beam_width):
                        decoded_t = indices[:, :, new_k].squeeze(0)
                        log_p = log_prob[:, :, new_k].item()

                        node = BeamSearchNode(
                            hidden_state=deepcopy(current_hidden),
                            prev_node=n,
                            token_id=deepcopy(decoded_t),
                            log_prob=n.logp+log_p,
                            length=n.len+1,
                        )
                        score = -node.eval()
                        next_nodes.append((score, node))
                    
                    for i in range(len(next_nodes)):
                        score, next_node = next_nodes[i]
                        nodes.put((score, next_node))

                    num_steps += beam_width
                
                # <EOS> 토큰이 한번도 등장하지 않았을 경우 - 최대 확률 노드
                if len(end_nodes) == 0:
                    end_nodes = [nodes.get() for _ in range(topk)]

                utterances = []
                for score, n in sorted(
                    end_nodes, key=operator.itemgetter(0)
                ):  # 가장 마지막 노드에서 역추적
                    utterance = []
                    utterance.append(n.token_id.item())
                    # back trace
                    while n.prev_node != None:
                        n = n.prev_node
                        utterance.append(n.token_id.item())

                    utterance = utterance[::-1]  # 뒤집기
                    utterances.append(utterance)

                if topk == 1:
                    decoded_batch.append(utterances[0])
                else:
                    decoded_batch.append(utterances)

        # id_to_string의 입력에 맞게 텐서로 변경
        outputs = []
        for decoded_sample in decoded_batch:
            if len(decoded_sample) < max_sequence:
                num_pads = max_sequence - len(decoded_sample)
                decoded_sample += [pad_token_id]*num_pads
            elif len(decoded_sample) > max_sequence:
                decoded_sample = decoded_sample[:max_sequence]
            outputs.append(decoded_sample)
        outputs = torch.tensor(outputs)

        return outputs



# just for debug
if __name__ == '__main__':
    from flags import Flags
    from dataset import dataset_loader
    from train import get_train_transforms, get_valid_transforms
    from utils import set_seed
    CONFIG_PATH = "./configs/SATRN-dev.yaml"
    options = Flags(CONFIG_PATH).get()
    set_seed(seed=options.seed)

    # get data
    (
        train_data_loader,
        validation_data_loader,
        train_dataset,
        valid_dataset,
    ) = dataset_loader(
        options=options,
        train_transform=get_train_transforms(
            options.input_size.height, options.input_size.width
        ),
        valid_transform=get_valid_transforms(
            options.input_size.height, options.input_size.width
            ),
            )

    model = SATRN(options, train_dataset)
    model.cuda()
    model.eval()
    batch = next(iter(train_data_loader))
    input = batch['image'].float().to(device)

    model.beam_search(input=input, data_loader=train_data_loader)