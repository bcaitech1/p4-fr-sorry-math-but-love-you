import torch
from torch.utils.data import DataLoader
from queue import PriorityQueue


def decode(
    model,
    input: torch.Tensor, 
    data_loader: DataLoader=None, 
    expected: torch.Tensor=None, 
    decode_type: str='greedy', 
    beam_width: int=3
    ) -> torch.Tensor:
    """디코딩을 수행하는 함수. NOTE: inference/validation에만 활용!

    Args:
        model (torch.nn.Module):
        input (torch.Tensor): 이미지 input
        data_loader (DataLoader, optional):
            - Beam-Search를 활용할 경우 필요한 argument
            - Defaults to None.
        expected (torch.Tensor, optional):
            - Validation에서 디코딩할 경우 필요한 argument: Validation 단계에서는 max_length를 ground-truth를 바탕으로 설정하기 때문
            - Defaults to None.
        decode_type (str, optional): 디코딩 타입 설정. Defaults to 'greedy'.
            - 'greedy': 그리디 디코딩
            - 'beam': 빔서치
        beam_width (int, optional): [description]. Defaults to 3.
            - 빔서치 활용 시 채택할 beam size

    Returns:
        squence (torch.Tensor): id_to_string에 입력 가능한 output sequence 텐서
    """

    if decode_type == 'greedy':
        output = model(
            input=input, 
            expected=expected, 
            is_train=False,
            teacher_forcing_ratio=0.0
            )
        decoded_values = output.transpose(1, 2) # [B, VOCAB_SIZE, MAX_LEN]
        _, sequence = torch.topk(decoded_values, 1, dim=1) # sequence: [B, 1, MAX_LEN]
        sequence = sequence.squeeze(1) # [B, MAX_LEN], 각 샘플에 대해 시퀀스가 생성 상태

    elif decode_type == 'beam':
        sequence = model.beam_search(
            input=input,
            data_loader=data_loader,
            beam_width=beam_width,
            max_sequence=expected.size(-1)-1 # expected에는 이미 시작 토큰 개수까지 포함
        )
    else:
        raise NotImplementedError

    return sequence


class BeamSearchNode(object):
    def __init__(self, hidden_state, prev_node, token_id, log_prob, length):
        self.hidden_state = hidden_state
        self.prev_node = prev_node
        self.token_id = token_id
        self.logp = log_prob
        self.len = length  # 길이

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.len - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.len < other.len

    def __gt__(self, other):
        return self.len > other.len