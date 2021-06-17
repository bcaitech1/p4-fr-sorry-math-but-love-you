import sys
import math
import random
import operator
from queue import PriorityQueue
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import timm
import sys

sys.path.append("../")
from postprocessing.decoding import BeamSearchNode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepCNN(nn.Module):
    def __init__(self, nc: int, leaky_relu=False):
        super(DeepCNN, self).__init__()
        m = timm.create_model("tf_efficientnetv2_s_in21ft1k", pretrained=True)
        self.conv_stem = nn.Conv2d(
            nc, 24, kernel_size=(3, 3), stride=(2, 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            24, eps=1e-3, momentum=0.1, affine=True, track_running_stats=True
        )
        self.act1 = nn.SiLU(inplace=True)
        self.eff_blocks = m.blocks
        self.pooling1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        self.conv1 = self.convRelu(nc, 4, leaky_relu=leaky_relu, bn=True)
        self.conv2 = self.convRelu(nc, 5, leaky_relu=leaky_relu)
        self.pooling2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        self.conv3 = self.convRelu(nc, 6, leaky_relu=leaky_relu, bn=True)

    def forward(self, input):
        out = self.conv_stem(input)  # [B, 24, 127, 511]
        out = self.bn1(out)  # [B, 24, 127, 511]
        out = self.act1(out)  # [B, 24, 127, 511]
        out = self.eff_blocks(out)  # [B, 256, 8, 32]
        out = self.pooling1(out)  # [B, 256, 4, 33]
        out = self.conv1(out)  # [B, 384, 4, 33]
        out = self.pooling2(out)  # [B, 384, 2, 34]
        out = self.conv3(out)  # [B, 384, 1, 33]
        return out

    @staticmethod
    def convRelu(nc, i, leaky_relu: bool, bn=False) -> nn.Module:
        """Conv 레이어를 생성하는 함수
        Args:
            i (int): ks, ps, ss, nm으로부터 가져올 element의 인덱스
            batchNormalization (bool, optional): BN 적용 여부. Defaults to False.
        """
        ks = [3, 3, 3, 3, 3, 3, 2]  # kernel size
        ps = [1, 1, 1, 1, 1, 1, 0]  # padding size
        ss = [1, 1, 1, 1, 1, 1, 1]  # strides
        # nm = [64, 128, 256, 256, 512, 512, 512]  # output channel list
        nm = [64, 128, 256, 256, 384, 384, 384]  # output channel list

        cnn = nn.Sequential()
        nIn = nc if i == 0 else nm[i - 1]
        nOut = nm[i]
        cnn.add_module("conv{0}".format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
        if bn:
            cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(nOut))

        if leaky_relu:
            cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2, inplace=True))
        else:
            cnn.add_module("relu{0}".format(i), nn.ReLU(True))

        return cnn


class AttentionCell(nn.Module):
    """디코더(ASTERDecoder) 내 Attention 계산에 활용할 attention cell 클래스"""

    def __init__(
        self,
        src_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers=1,
    ):
        """
        Args:
            src_dim (int): 입력 데이터의 dim
            hidden_dim (int):
                - RNN 내에서 활용될 hidden state의 dim
            embedding_dim (int):
                - 입력 데이터의 임베딩 결과 dim
                - 입력 데이터가 RNN 레이어에 입력되기 전에 임베딩 과정을 거침
            num_layers (int, optional): RNN 레이어 수. Defaults to 1.
            cell_type (str, optional): RNN 모델 타입. Defaults to 'LSTM'
                - RNN의 input dim: src_dim + embedding_dim
        """
        super(AttentionCell, self).__init__()
        self.num_layers = num_layers

        self.i2h = nn.Linear(src_dim, hidden_dim, bias=False)  # input dim to hidden dim

        # hidden dim to hidden dim
        self.h2h = nn.Linear(
            hidden_dim, hidden_dim
        )  # either i2i or h2h should have bias <- why?
        self.score = nn.Linear(hidden_dim, 1, bias=False)  # to get attention logit

        if num_layers == 1:
            self.rnn = nn.LSTMCell(src_dim + embedding_dim, hidden_dim)
        else:
            self.rnn = nn.ModuleList(
                [nn.LSTMCell(src_dim + embedding_dim, hidden_dim)]
                + [nn.LSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
            )
        self.hidden_dim = hidden_dim

    def forward(self, prev_hidden, src, tgt):  # src: [b, L, c]
        """
        input:
            prev_hidden (torch.Tensor): 이전 state에서의 hidden state. [b, h]
            src (torch.Tensor): X_t. [b, L, c]
            tgt (torch.Tensor): START 토큰 등 RNN의 입력으로 들어가는 텐서
        output:
        """
        src_features = self.i2h(src)  # [b, L, h]
        if self.num_layers == 1:
            prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)  # [b, 1, h]
        else:
            prev_hidden_proj = self.h2h(prev_hidden[-1][0]).unsqueeze(1)  # [b, 1, h]
        attention_logit = self.score(
            torch.tanh(src_features + prev_hidden_proj)  # [b, L, h]
        )  # [b, L, 1]
        alpha = F.softmax(attention_logit, dim=1)  # [b, L, 1]
        context = torch.bmm(alpha.permute(0, 2, 1), src).squeeze(
            1
        )  # [b, c], values applied attention

        concat_context = torch.cat([context, tgt], 1)  # [b, c+e]

        # cur_hidden: [hidden_state, cell_state] for LSTM
        if self.num_layers == 1:
            cur_hidden = self.rnn(concat_context, prev_hidden)
        else:
            cur_hidden = []
            for i, layer in enumerate(self.rnn):
                if i == 0:
                    concat_context = layer(concat_context, prev_hidden[i])
                else:
                    concat_context = layer(concat_context[0], prev_hidden[i])
                cur_hidden.append(concat_context)

        return cur_hidden, alpha


class ASTEREncoder(nn.Module):
    def __init__(self, FLAGS):
        super(ASTEREncoder, self).__init__()
        self.cnn = DeepCNN(nc=FLAGS.data.rgb)
        self.blstm = nn.LSTM(
            input_size=384,  # H*C = 256*7
            hidden_size=FLAGS.ASTER.hidden_dim,
            num_layers=2,
            bidirectional=True,
        )
        self.proj = nn.Linear(
            in_features=FLAGS.ASTER.hidden_dim * 2, out_features=FLAGS.ASTER.hidden_dim
        )

    def forward(self, input):
        out = self.cnn(input)  # [B, SEQ_LEN, HIDDEN]
        b, c, h, w = out.size()
        out = out.view(b, c * h, w).permute(2, 0, 1)  # [SEQ_LEN, B, INPUT_SIZE]
        out, _ = self.blstm(out)  # [SEQ_LEN, B, HIDDEN*2]
        out = self.proj(out)  # [SEQ_LEN, B, HIDDEN]
        out = out.transpose(0, 1)  # [B, SEQ_LEN, HIDDEN]
        return out


class ASTERDecoder(nn.Module):
    def __init__(
        self,
        num_classes,
        src_dim,
        embedding_dim,
        hidden_dim,
        pad_id,
        st_id,
        num_layers=1,
        checkpoint=None,
        decoding_manager=None,
    ):
        super(ASTERDecoder, self).__init__()
        self.embedding = nn.Embedding(num_classes + 1, embedding_dim)
        self.attention_cell = AttentionCell(
            src_dim, hidden_dim, embedding_dim, num_layers
        )
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.generator = nn.Linear(hidden_dim, num_classes)
        self.pad_id = pad_id
        self.st_id = st_id
        self.manager = decoding_manager

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def forward(
        self, src, text, is_train=True, teacher_forcing_ratio=1.0, batch_max_length=50
    ):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [START] token. text[:, 0] = [START].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = src.size(0)
        num_steps = batch_max_length - 1  # +1 for [s] at end of sentence.

        # LSTM case
        if self.num_layers == 1:
            hidden = (
                torch.FloatTensor(batch_size, self.hidden_dim)
                .fill_(0)
                .to(device),  # hidden
                torch.FloatTensor(batch_size, self.hidden_dim)
                .fill_(0)
                .to(device),  # cell
            )
        else:
            hidden = [
                (
                    torch.FloatTensor(batch_size, self.hidden_dim)
                    .fill_(0)
                    .to(device),  # hidden
                    torch.FloatTensor(batch_size, self.hidden_dim)
                    .fill_(0)
                    .to(device),  # cell
                )
                for _ in range(self.num_layers)
            ]

        # teacher forcing할 경우
        if is_train:
            if random.random() < teacher_forcing_ratio:
                # hidden 만들어놓고
                output_hiddens = (
                    torch.FloatTensor(batch_size, num_steps, self.hidden_dim)
                    .fill_(0)
                    .to(device)
                )
                # 배치 내 모든 sample에 대해 일괄 계산
                for i in range(num_steps):
                    embedd = self.embedding(text[:, i])  # 샘플 각각의 i번째 캐릭터 임베딩
                    hidden, alpha = self.attention_cell(
                        prev_hidden=hidden,  # [hidden_state, cell_state] for LSTM, [B, HIDDEN]
                        src=src,  # [B, WxH, C]
                        tgt=embedd,  # [B, HIDDEN]
                    )  # hidden: [B, HIDDEN] x2
                    if self.num_layers == 1:
                        output_hiddens[:, i, :] = hidden[
                            0
                        ]  # for LSTM (0: hidden, 1: Cell)
                    else:
                        output_hiddens[:, i, :] = hidden[-1][0]  # 다중 레이어 - 마지막 레이어 사용
                probs = self.generator(output_hiddens)

            else:
                # [SOS] - [B]
                targets = torch.LongTensor(batch_size).fill_(self.st_id).to(device)
                probs = (
                    torch.FloatTensor(batch_size, num_steps, self.num_classes)
                    .fill_(0)
                    .to(device)
                )  # Prob Table - [B, MAX_LEN, VOCAB_SIZE]

                for i in range(num_steps):
                    embedd = self.embedding(targets)  # [B, HIDDEN]
                    hidden, alpha = self.attention_cell(
                        prev_hidden=hidden,  # [hidden_state, cell_state] for LSTM, [B, HIDDEN]
                        src=src,  # [B, WxH, C]
                        tgt=embedd,  # [B, HIDDEN]
                    )  # hidden: [B, HIDDEN] x2
                    probs_step = (
                        self.generator(hidden[0])
                        if self.num_layers == 1
                        else self.generator(hidden[-1][0])
                    )  # [B, VOCAB_SIZE]

                    probs[:, i, :] = probs_step  # step_{i}에 추가. 실제로는 소프트맥스 이전 값이 들어감
                    _, next_input = probs_step.max(1)  # next_input: [B](=targets 사이즈)
                    targets = next_input  # 이전 스텝 출력을 현재 스텝 입력으로

        else:
            targets = torch.LongTensor(batch_size).fill_(self.st_id).to(device)
            probs = (
                torch.FloatTensor(batch_size, num_steps, self.num_classes)
                .fill_(0)
                .to(device)
            )  # Prob Table - [B, MAX_LEN, VOCAB_SIZE]
            if self.manager is not None:
                self.manager.reset(sequence_length=num_steps)

            for i in range(num_steps):
                embedd = self.embedding(targets)  # [B, HIDDEN]
                hidden, alpha = self.attention_cell(
                    prev_hidden=hidden,  # [hidden_state, cell_state] for LSTM, [B, HIDDEN]
                    src=src,  # [B, WxH, C]
                    tgt=embedd,  # [B, HIDDEN]
                )  # hidden: [B, HIDDEN] x2
                probs_step = (
                    self.generator(hidden[0])
                    if self.num_layers == 1
                    else self.generator(hidden[-1][0])
                )  # [B, VOCAB_SIZE]

                # NOTE: DecodingManager
                if self.manager is not None:
                    next_input, probs_step = self.manager.sift(probs_step)
                    probs[:, i, :] = probs_step  # step_{i}에 추가. 실제로는 소프트맥스 이전 값이 들어감
                    targets = next_input
                else:
                    probs[:, i, :] = probs_step  # step_{i}에 추가. 실제로는 소프트맥스 이전 값이 들어감
                    _, next_input = probs_step.max(1)  # next_input: [B](=targets 사이즈)
                    targets = next_input  # 이전 스텝 출력을 현재 스텝 입력으로

        return probs


class ASTER(nn.Module):
    def __init__(
        self, FLAGS, train_dataset, checkpoint=None, decoding_manager=None  # NOTE
    ):
        super(ASTER, self).__init__()
        self.encoder = ASTEREncoder(FLAGS)
        self.decoder = ASTERDecoder(
            num_classes=len(train_dataset.id_to_token),
            src_dim=FLAGS.ASTER.src_dim,
            embedding_dim=FLAGS.ASTER.embedding_dim,
            hidden_dim=FLAGS.ASTER.hidden_dim,
            pad_id=train_dataset.token_to_id["<PAD>"],
            st_id=train_dataset.token_to_id["<SOS>"],
            num_layers=FLAGS.ASTER.layer_num,
            decoding_manager=decoding_manager,  # NOTE
        )

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=train_dataset.token_to_id["<PAD>"]
        )

        if checkpoint:
            self.load_state_dict(checkpoint)

    def forward(self, input, expected, is_train, teacher_forcing_ratio):
        out = self.encoder(input)  # [b, C, H, W]
        output = self.decoder(
            src=out,
            text=expected,
            is_train=is_train,
            teacher_forcing_ratio=teacher_forcing_ratio,
            batch_max_length=expected.size(1),
        )  # [B, MAX_LEN, VOCAB_SIZE]
        return output

    def beam_search(
        self,
        input: torch.Tensor,
        data_loader: DataLoader,
        topk: int = 1,  # 상위 몇 개의 결과를 얻을 것인지
        beam_width: int = 10,  # 각 스텝마다 몇 개의 후보군을 선별할지
        max_sequence: int = 230,
    ):
        """빔서치 디코딩을 수행하는 함수. inference시에만 활용
        Args:
            input (Tensor): DataLoader로부터 얻은 input batch
            data_loader (DataLoader): inference에 활용되는 DataLoader. 스페셜 토큰 ID 정보를 얻기 위함
            beam_width(int): 스텝마다 확률이 높게 측정된 토큰을 상위 몇 개 추릴 것인지 설정
                - Greedy Decoding에 비해 (beam_width)배 만큼 추론 시간 소요
            topk (int, optional): 각 샘플 당 몇 개의 문장을 생성할 지. Defaults to 1
                - NOTE. 현재 top1에 대해서만 구현됨
            max_sequence(int): 최대 몇 step까지 생성할지 설정
        Returns:
            Tensor: id_to_string에 입력 가능한 형태의 텐서 [B, MAX_SEQUENCE]
        References.
            - budzianowski, PyTorch-Beam-Search, https://github.com/budzianowski/PyTorch-Beam-Search-Decoding
            - 312shaun, Pytorch-seq2seq-Beam-Search, https://github.com/312shan/Pytorch-seq2seq-Beam-Search
        """
        sos_token_id = data_loader.dataset.token_to_id["<SOS>"]
        eos_token_id = data_loader.dataset.token_to_id["<EOS>"]
        pad_token_id = data_loader.dataset.token_to_id["<PAD>"]

        batch_size = len(input)
        src = self.encoder(input)

        decoded_batch = []
        with torch.no_grad():
            hiddens = self._initialize_hidden_states(
                batch_size=batch_size
            )  # [B, HIDDEN]x2

            # 문장 단위 생성
            for data_idx in range(batch_size):

                end_nodes = []
                number_required = min((topk + 1), topk - len(end_nodes))  # 최대 생성 횟수

                # 빔서치 과정 상 역추적을 위한 우선순위큐 선언
                nodes = PriorityQueue()

                # 시작 토큰 초기화
                current_src = src[data_idx, :, :].unsqueeze(0)  # [B=1, W=33, C=384]
                current_input = torch.LongTensor([sos_token_id])  # [1]
                current_hiddens = [
                    (h[data_idx, :].unsqueeze(0), c[data_idx, :].unsqueeze(0))
                    for (h, c) in hiddens
                ]
                node = BeamSearchNode(
                    hidden_state=deepcopy(current_hiddens),
                    prev_node=None,
                    token_id=deepcopy(current_input),
                    log_prob=0,
                    length=1,
                )
                score = -node.eval()

                # 최대힙: 확률 높은 토큰을 추출하기 위함
                nodes.put((score, node))

                num_steps = 0
                while True:
                    if num_steps >= (max_sequence - 1) * beam_width:
                        break

                    # 최대확률샘플 추출/제거, score: 로그확률, n: BeamSearchNode
                    score, n = nodes.get()
                    current_input = n.token_id  # 토큰 ID # [B=1]
                    current_hiddens = n.hidden_state  # hidden state

                    # 종료 토큰이 생성될 경우(종료 토큰 & 이전 노드 존재)
                    if n.token_id.item() == eos_token_id and n.prev_node != None:
                        end_nodes.append((score, n))
                        if len(end_nodes) >= number_required:
                            break
                        else:
                            continue

                    # ------디코딩------
                    # input_embedded = self.decoder.embedding(current_input.to(input.get_device()))
                    input_embedded = self.decoder.embedding(current_input.to(device))

                    # Hidden state 갱신
                    current_hiddens, alpha = self.decoder.attention_cell(
                        prev_hidden=current_hiddens, src=current_src, tgt=input_embedded
                    )  # hidden state 갱신
                    prob_step = self.decoder.generator(
                        current_hiddens[-1][0]
                    )  # [1, VOCAB_SIZE] (num_layers=1) ***앙상블에 필요한 로짓

                    # 모델의 로짓을 확률화
                    log_prob_step = F.log_softmax(prob_step, dim=-1)  # [1, VOCAB_SIZE]
                    log_prob, indices = torch.topk(log_prob_step, beam_width)

                    # 다음 state에 활용할 {beam_width}개 후보 노드를 put
                    next_nodes = []
                    for new_k in range(beam_width):
                        decoded_t = indices[0][new_k].view(-1)
                        log_p = log_prob[0][new_k].item()
                        node = BeamSearchNode(
                            hidden_state=deepcopy(current_hiddens),
                            prev_node=n,
                            token_id=deepcopy(decoded_t),
                            log_prob=n.logp + log_p,
                            length=n.len + 1,
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
                for score, n in sorted(end_nodes, key=operator.itemgetter(0)):
                    utterance = []
                    utterance.append(n.token_id.item())
                    # 가장 마지막 노드에서 역추적
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
                decoded_sample += [pad_token_id] * num_pads
            elif len(decoded_sample) > max_sequence:
                decoded_sample = decoded_sample[:max_sequence]
            outputs.append(decoded_sample)
        outputs = torch.tensor(outputs)

        return outputs

    def _initialize_hidden_states(self, batch_size: int):
        if self.decoder.num_layers == 1:
            hidden = (
                torch.FloatTensor(batch_size, self.decoder.hidden_dim)
                .fill_(0)
                .to(device),  # hidden
                torch.FloatTensor(batch_size, self.decoder.hidden_dim)
                .fill_(0)
                .to(device),  # cell
            )
        else:
            hidden = [
                (
                    torch.FloatTensor(batch_size, self.decoder.hidden_dim)
                    .fill_(0)
                    .to(device),  # hidden
                    torch.FloatTensor(batch_size, self.decoder.hidden_dim)
                    .fill_(0)
                    .to(device),  # cell
                )
                for _ in range(self.decoder.num_layers)
            ]
        return hidden


class ASTER_encoder(nn.Module):
    def __init__(self, FLAGS, checkpoint=None):
        super(ASTER_encoder, self).__init__()
        self.encoder = ASTEREncoder(FLAGS)
        if checkpoint:
            self.load_state_dict(checkpoint)

    def forward(self, input):
        enc_result = self.encoder(input)
        return enc_result


class ASTER_decoder(nn.Module):
    def __init__(self, FLAGS, train_dataset, checkpoint=None):
        super(ASTER_decoder, self).__init__()
        self.decoder = ASTERDecoder(
            num_classes=len(train_dataset.id_to_token),
            src_dim=FLAGS.ASTER.src_dim,
            embedding_dim=FLAGS.ASTER.embedding_dim,
            hidden_dim=FLAGS.ASTER.hidden_dim,
            pad_id=train_dataset.token_to_id["<PAD>"],
            st_id=train_dataset.token_to_id["<SOS>"],
            num_layers=FLAGS.ASTER.layer_num,
        )
        self.step_idx = 0
        self.hidden = None
        if checkpoint:
            self.load_state_dict(checkpoint)

    def forward(
        self,
        src,
        expected,
        is_train=False,
        teacher_forcing_ratio=0.0,
        batch_max_length=50,
    ):
        dec_result = self.decoder(
            src=src,
            text=expected,
            is_train=is_train,
            teacher_forcing_ratio=teacher_forcing_ratio,
            batch_max_length=expected.size(1),
        )
        return dec_result

    def step_forward(self, src, target) -> torch.Tensor:
        if self.hidden is None:
            batch_size = len(src)
            self.hidden = self._initialize_hidden_states(batch_size)

        embedd = self.decoder.embedding(target)
        self.hidden, _ = self.decoder.attention_cell(
            prev_hidden=self.hidden, src=src, tgt=embedd
        )
        _out = (
            self.decoder.generator(self.hidden[0])
            if self.decoder.num_layers == 1
            else self.decoder.generator(self.hidden[-1][0])
        )  # [B, VOCAB_SIZE]
        self.step_idx += 1
        return _out

    def reset_status(self):
        self.step_idx = 0
        self.hidden = None

    def _initialize_hidden_states(self, batch_size: int):
        if self.decoder.num_layers == 1:
            hidden = (
                torch.FloatTensor(batch_size, self.decoder.hidden_dim)
                .fill_(0)
                .to(device),  # hidden
                torch.FloatTensor(batch_size, self.decoder.hidden_dim)
                .fill_(0)
                .to(device),  # cell
            )
        else:
            hidden = [
                (
                    torch.FloatTensor(batch_size, self.decoder.hidden_dim)
                    .fill_(0)
                    .to(device),  # hidden
                    torch.FloatTensor(batch_size, self.decoder.hidden_dim)
                    .fill_(0)
                    .to(device),  # cell
                )
                for _ in range(self.decoder.num_layers)
            ]
        return hidden
