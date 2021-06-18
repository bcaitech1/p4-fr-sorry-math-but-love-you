# NOTE: 수정 - 사소한 리패터링만 수행. 복붙 ㄱㄱ!
from copy import deepcopy
import math
import warnings
from typing import List
import torch
import torch.nn.functional as F
import sys

from data import SPECIAL_TOKENS

RULES = {
    # 한번도 첫번째로 등장한 적이 없는 토큰
    "cannot_initial": [
        "\\downarrow", "\\supsetneq", "\\subset", "\\right|", "\\not", "\\roman1", "\\cong", 
        "\\rightleftarrows", "\\wedge", "!", "\\prime", "?", "\\right.", "\\vdots", "\\end{matrix}", 
        "\\uparrow", "\\\\", "\\perp", "\\nsubseteq", "\\Pi", "\\subsetneq", "\\notin", 
        "\\backslash", "\\nexists", "\\frown", "\\right\\|", "\\doteq", "\\searrow", "\\propto", 
        "\\vee", "\\supset", "\\Theta", "\\roman2", "\\geqq", "\\subseteq", "%", "\\odot", 
        "\\oplus", "\\simeq", "\\ni", "\\omicron", "}", "\\Omega", "\\roman5",
    ],
    # 매우 높은 확률로 뒤에 '_' 토큰이 붙는 토큰 # NOTE 성능 저하를 유발해 제거
    "next_underbar": [
        # '\\lim', '\\iint'
    ],
    # 매우 높은 확률로 뒤에 '{' 토큰이 붙는 토큰 # NOTE 성능 저하를 유발해 제거
    "next_lbracket": [
        # '^', '_', '\\sqrt', '\\frac', '\\overline'
    ],
    # 매우 높은 확률로 뒤에 '_' 토큰이 붙지 않는 토큰 # NOTE 성능 저하를 유발해 제거
    "cannot_next_underbar": [
        # NOTE: V3
        # "'", '\\right)', '\\alpha', '\\gamma', '\\pi', '/', '{', '|',
        # '=', '\\right|', '\\Rightarrow', '\\left(', '\\times', ':', '\\cup',
        # '\\cap', '!', '\\Delta', '\\forall', '\\cdot', '>', '\\nu', '\\infty', '\\', '+',
        # ',', '\\prime', '\\ln', '\\therefore', '\\right.',
        # NOTE: V2
        # "'", '\\right)', '\\alpha', '\\gamma', '\\pi', '/', '{', '|',
        # '=', '\\right|', '\\Rightarrow', '\\left(', '\\times', ':', '\\cup',
        # '\\cap', '!', '\\Delta', '\\forall', '\\cdot', '>', '\\nu', '\\infty', '\\', '+',
        # ',', '\\prime', '\\ln', '\\therefore', '\\right.',
        # NOTE: V1
        # '}', "'", '\\right)', '\\alpha', '\\gamma', '\\pi', '/', '{', '|',
        # '=', '\\right|', '\\Rightarrow', '\\left(', '\\times', ':', '\\cup',
        # '\\cap', '!', '\\Delta', '\\forall', '\\cdot', '>', '\\nu', '\\infty', '\\', '+',
        # ',', '\\prime', '\\ln', '\\therefore', '\\right.',
        # NOTE: Origin
        # '}', 'q', 'B', 'r', 'f', 'b', 'l', 'k', 't', 'A', 'j', "'", 'c', '\\right)', 'd', 'p',
        # '\\alpha', '\\gamma', '3', 's', 'h', '8', 'g', 'n', 'w', 'e', '\\pi', '/', 'D', '{', '|',
        # '=', '\\right|', '\\Rightarrow', '4', '\\left(', '\\times', '5', ':', '0', '2', '\\cup',
        # '\\cap', '!', '1', '\\Delta', '\\forall', '\\cdot', '>', '\\nu', '\\infty', '\\', '+',
        # '6', 'O', 'i', ',', '\\prime', '\\ln', '9', '\\therefore', '\\right.'
    ],
    # 매우 높은 확률로 뒤에 '{' 토큰이 붙지 않는 토큰
    "cannot_next_lbracket": [
        # NOTE: V1
        # '=', '\\tan', '\\right)', ',', ':', '\\therefore', '+', '\\sin',
        # '\\leq', '\\left(', '\\int', '\\begin{matrix}', '\\sum', '\\ln', '\\tau', '\\cdot',
        # '\\to', '\\rightarrow', '\\circ', '>', '\\cos', '\\pi', '\\log', '\\left[', '\\csc',
        # '\\in', '\\subset', '\\times', '\\neq', '\\right.', '\\cap', '?', '\\cdots',
        # '\\sim', '-', '\\geq', '\\left|', '\\phi', '/', '\\Leftrightarrow', '\\leqq',
        # '|', '\\epsilon', '\\nabla', '<', '\\left.', '\\Pi', '\\left\\{', '.',
        # '\\pm', '\\beta', '\\right\\}', '\\omega', '!', '\\cup', '\\forall',
        # '\\right|', '\\notin', '\\mu', '\\approx', '\\cot', '\\left\\|', '\\square',
        # NOTE: Origin
        # '7', '=', '0', '2', '\\tan', '\\right)', ',', ':', '\\therefore', '8', '5', '+', '\\sin',
        # '\\leq', '6', '\\left(', '\\int', '\\begin{matrix}', '\\sum', 'p', '\\ln', '\\tau', '\\cdot',
        # '\\to', 'x', '\\rightarrow', '\\circ', '>', '\\cos', '\\pi', '\\log', 'A', '\\left[', '\\csc',
        # '\\in', '1', 'y', '\\subset', '\\times', '\\neq', '\\right.', '\\cap', '?', '\\cdots', 'm',
        # '4', '\\sim', 'k', '-', '\\geq', '\\left|', 'Y', 'n', '\\phi', '/', '\\Leftrightarrow', '\\leqq',
        # '|', 'C', '\\epsilon', '\\nabla', '<', 'f', 'g', '\\left.', '\\Pi', '&', '\\left\\{', '.',
        # '\\pm', 'd', '\\beta', '\\right\\}', 'b', '3', 'a', '9', 'S', '\\omega', '!', '\\cup', '\\forall',
        # 'r', '\\right|', '\\notin', '\\mu', 'l', 'B', '\\approx', '\\cot', 'z', '\\left\\|', 's', '\\square'
    ],
    # 최대 연속 생성 횟수를 규제할 토큰
    # NOTE. 숫자와 소문자 알파벳은 포함하지 않음
    # 연속 생성 횟수를 규제할 토큰인지 여부
    "limit_series": {
        "<PAD>": False, "<EOS>": False, "O": True, "\\prod": True, "\\downarrow": True, "\\widehat": True, 
        "\\iiint": True, "\\ddot": True, "\\supsetneq": True, "_": True, "\\log": True, "\\dot": True, 
        "\\sqrt": True, "0": False, "6": False, "\\ominus": True, ";": True, "g": False, "\\leqq": True, 
        "I": True, "\\tanh": True, "\\subset": True, "\\pm": True, "\\oint": True, "f": False,
        "\\rightharpoonup": True, "\\right|": True, "\\tau": True, "\\underset": True, "\\not": True, 
        "<": True, "\\cos": True, "\\theta": True, "\\rho": True, "\\epsilon": True, "\\roman1": True, 
        "\\ln": True, "\\cong": True, "/": True, "X": True, "b": False, "\\sec": True, "K": True, 
        "\\zeta": True, "V": True, "\\rightleftarrows": True, "\\wedge": True, "!": True, "\\prime": True, 
        "\\left|": True, "\\nabla": True, "Z": True, "\\stackrel": True, "M": True, "?": True, "v": False, 
        "P": True, "\\leq": True, "\\cdots": True, "\\cup": True, "n": False, "Q": True, "\\square": True, 
        "\\kappa": True, "\\exists": True, "'": True, "\\gamma": True, "\\in": True, "\\eta": True, "5": False, 
        "\\right.": True, "\\vdots": True, "\\left\\|": True, "\\end{matrix}": True, "\\uparrow": True, 
        "N": True, "\\neq": True, "|": True, "\\phi": True, "\\Rightarrow": True, "\\\\": True, "\\frac": True, 
        "\\because": True, "\\sim": True, "4": False, "\\perp": True, "j": False, "\\nsubseteq": True, 
        "L": True, "A": True, "\\Psi": True, "w": False, ",": True, "\\cosec": True, "\\fallingdotseq": True, 
        "D": True, "\\Pi": True, "&": True, "y": False, "\\sinh": True, "\\beta": True, "\\pi": True, 
        "\\nu": True, "$": True, "\\subsetneq": True, "\\forall": True, "\\tan": True, "+": True, 
        "\\notin": True, "\\xi": True, "3": False, "\\upsilon": True, "\\rightarrow": True, "\\angle": True, 
        "\\backslash": True, "G": True, "\\sum": True, "\\nexists": True, "\\frown": True, ">": True, 
        "\\div": True, "\\cot": True, "t": False, "\\right\\|": True, "1": False, "\\overrightarrow": True, 
        "*": True, "\\Gamma": True, "\\doteq": True, "i": True, "\\varnothing": True, "\\mp": True, '"': True, 
        "z": False, "Y": True, "8": False, "\\right)": True, "\\delta": True, "\\leftarrow": True, "u": False, 
        "9": False, "\\overleftrightarrow": True, "\\searrow": True, "q": False, "\\right\\}": True, 
        "\\equiv": True, "J": True, "\\geq": True, "m": False, "\\propto": True, "R": True, "\\vee": True, 
        "\\sigma": True, "\\cap": True, "\\supset": True, "\\Theta": True, "p": False, "\\sin": True, "C": True, 
        "7": False, "\\partial": True, "\\csc": True, "\\to": True, "r": False, ":": True, "H": True, 
        "\\hbar": True, "\\lim": True, "k": False, "\\cdot": True, "\\left[": True, "\\left.": True, 
        "\\Lambda": True, "\\begin{matrix}": True, "\\left\\{": True, "h": False, "2": False, "F": True, 
        "s": False, "=": True, "T": True, ".": True, "\\mu": True, "\\roman2": True, "S": True, "\\alpha": True, 
        "\\geqq": True, "W": True, "\\int": True, "\\right]": True, "\\therefore": True, "\\subseteq": True, 
        "%": True, "\\leftrightarrow": True, "\\csch": True, "\\overline": True, "\\odot": True, "\\varphi": True, 
        "\\times": True, "\\min": True, "\\iint": True, "\\oplus": True, "\\widetilde": True, "x": True, 
        "a": False, "U": True, "\\simeq": True, "o": False, "\\ni": True, "\\omicron": True, "\\Delta": True, 
        "\\Leftarrow": True, "}": True, "-": True, "d": False, "\\lambda": True, "\\varrho": True, 
        "\\Leftrightarrow": True, "\\roman4": True, "\\": True, "\\Phi": True, "\\psi": True, "\\omega": True, 
        "{": True, "\\coth": True, "\\sech": True, "\\Omega": True, "\\roman5": True, "\\left(": True, 
        "\\triangle": True, "\\overarc": True, "\\circ": True, "\\infty": True, "c": False, "E": True, "^": True, 
        "B": True, "\\otimes": True, "\\approx": True, "l": False, "\\max": True, "\\cosh": True, "e": False,
    },
    # 규제할 토큰별 규제 파라미터
    "limit_params": {
        #  최대 5회 연속
        "\\cdot": 5, ".": 5, "}": 5,
        # 최대 4회 연속
        "|": 4, "&": 4, "\\right)": 4, "x": 4,
        # 최대 3회 연속
        "\\prime": 3, "\\right.": 3, "A": 3, "i": 3, "\\int": 3,
        # 최대 2회 연속
        "O": 2, "\\right|": 2, "<": 2, "\\cos": 2, "/": 2, "!": 2, "\\left|": 2, "\\nabla": 2, "Z": 2, "P": 2, 
        "\\cdots": 2, "Q": 2, "'": 2, "\\gamma": 2, "\\in": 2, "\\\\": 2, ",": 2, "D": 2, "\\pi": 2, "+": 2, 
        ">": 2, "Y": 2, "\\right\\}": 2, "R": 2, "C": 2, ":": 2, "\\left[": 2, "\\left.": 2, "\\left\\{": 2, 
        "F": 2, "=": 2, "T": 2, "S": 2, "\\right]": 2, "\\times": 2, "-": 2, "\\": 2, "\\omega": 2, "{": 2, 
        "\\left(": 2, "E": 2, "B": 2,
        # 한번도 연속으로 등장한 적이 없는 토큰
        "\\prod": 1, "\\downarrow": 1, "\\widehat": 1, "\\iiint": 1, "\\ddot": 1, "\\supsetneq": 1, "_": 1, 
        "\\log": 1, "\\dot": 1, "\\sqrt": 1, "\\ominus": 1, ";": 1, "\\leqq": 1, "I": 1, "\\tanh": 1, 
        "\\subset": 1, "\\pm": 1, "\\oint": 1, "\\rightharpoonup": 1, "\\tau": 1, "\\underset": 1, "\\not": 1, 
        "\\theta": 1, "\\rho": 1, "\\epsilon": 1, "\\roman1": 1, "\\ln": 1, "\\cong": 1, "X": 1, "\\sec": 1, 
        "K": 1, "\\zeta": 1, "V": 1, "\\rightleftarrows": 1, "\\wedge": 1, "\\stackrel": 1, "M": 1, "?": 1, 
        "\\leq": 1, "\\cup": 1, "\\square": 1, "\\kappa": 1, "\\exists": 1, "\\eta": 1, "\\vdots": 1, 
        "\\left\\|": 1, "\\end{matrix}": 1, "\\uparrow": 1, "N": 1, "\\neq": 1, "\\phi": 1, "\\Rightarrow": 1, 
        "\\frac": 1, "\\because": 1, "\\sim": 1, "\\perp": 1, "\\nsubseteq": 1, "L": 1, "\\Psi": 1, "\\cosec": 1, 
        "\\fallingdotseq": 1, "\\Pi": 1, "\\sinh": 1, "\\beta": 1, "\\nu": 1, "$": 1, "\\subsetneq": 1, 
        "\\forall": 1, "\\tan": 1, "\\notin": 1, "\\xi": 1, "\\upsilon": 1, "\\rightarrow": 1, "\\angle": 1, 
        "\\backslash": 1, "G": 1, "\\sum": 1, "\\nexists": 1, "\\frown": 1, "\\div": 1, "\\cot": 1, "\\right\\|": 1, 
        "\\overrightarrow": 1, "*": 1, "\\Gamma": 1, "\\doteq": 1, "\\varnothing": 1, "\\mp": 1, '"': 1, 
        "\\delta": 1, "\\leftarrow": 1, "\\overleftrightarrow": 1, "\\searrow": 1, "\\equiv": 1, "J": 1, 
        "\\geq": 1, "\\propto": 1, "\\vee": 1, "\\sigma": 1, "\\cap": 1, "\\supset": 1, "\\Theta": 1, "\\sin": 1, 
        "\\partial": 1, "\\csc": 1, "\\to": 1, "H": 1, "\\hbar": 1, "\\lim": 1, "\\Lambda": 1, "\\begin{matrix}": 1, 
        "\\mu": 1, "\\roman2": 1, "\\alpha": 1, "\\geqq": 1, "W": 1, "\\therefore": 1, "\\subseteq": 1, "%": 1, 
        "\\leftrightarrow": 1, "\\csch": 1, "\\overline": 1, "\\odot": 1, "\\varphi": 1, "\\min": 1, "\\iint": 1, 
        "\\oplus": 1, "\\widetilde": 1, "U": 1, "\\simeq": 1, "\\ni": 1, "\\omicron": 1, "\\Delta": 1, 
        "\\Leftarrow": 1, "\\lambda": 1, "\\varrho": 1, "\\Leftrightarrow": 1, "\\roman4": 1, "\\Phi": 1, 
        "\\psi": 1, "\\coth": 1, "\\sech": 1, "\\Omega": 1, "\\roman5": 1, "\\triangle": 1, "\\overarc": 1, 
        "\\circ": 1, "\\infty": 1, "^": 1, "\\otimes": 1, "\\approx": 1, "\\max": 1, "\\cosh": 1,
    },
}


def get_decoding_manager(tokens_path: str, batch_size: int = 128):
    """DecodingManager를 구성하는 함수

    Args:
        tokens_path (str): DecodingManager 초기화에 활용할 토큰 경로
        batch_size (int, optional): 배치 사이즈. DecodingManager 내 MemoryNode 선언에 활용. Defaults to 128.
    """
    tokens = []
    tokens.extend(SPECIAL_TOKENS)
    with open(tokens_path, "r") as fd:
        reader = fd.read()
        for token in reader.split("\n"):
            if token not in tokens:
                tokens.append(token)
    assert len(tokens) == 245
    manager = DecodingManager(
        batch_size=batch_size,
        rules=RULES,
        tokens=tokens,
        # sequence_length=sequence_length,
    )
    return manager


class DecodingManager:
    def __init__(self, batch_size: int, rules: list, tokens: list):
        assert tokens[0] == "<SOS>", "It looks wrong tokens. Check again."
        self.tokens = tokens
        self.rules = rules
        self.batch_size = batch_size
        self.vocab_size = len(tokens)
        self.memories = None
        self.sequence_length = None

    def sift(self, probs_step: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs_step(torch.Tensor): softmax 확률 분포 텐서, [B, VOCAB_SIZE]
        Returns:
            targets(torch.Tensor): 다음 스텝에 입력될 target 텐서, [B]
        """
        dimension_flag = None  # SATRN과 ASTER의 probs_step 형태가 다르기 때문에 flag 마련

        # last batch 등 학습/추론 중 배치 사이즈가 달라질 경우 조정
        if len(probs_step) != self.batch_size:
            warnings.warn(
                f"batch size has been changed! {self.batch_size}->{len(probs_step)}"
            )
            self.memories = self._initialize_memories(
                len(probs_step), self.rules, self.tokens, self.sequence_length
            )
            self.batch_size = len(probs_step)

        if probs_step.ndim != 2:
            dimension_flag = 3
            probs_step = probs_step.squeeze(1).clone()

        mask = list(map(lambda x: self._mask(x, self.vocab_size), self.memories))
        mask = torch.vstack(mask)

        probs_softmax = F.softmax(probs_step, dim=-1)
        probs_softmax = probs_softmax.masked_fill(
            mask.to(probs_softmax.get_device()), 0
        )

        assert (probs_softmax == 0).sum().item() == mask.sum().item()
        targets = torch.argmax(probs_softmax, dim=-1)
        self._update(targets)

        if dimension_flag is not None:
            probs_softmax = probs_softmax.unsqueeze(1)
            assert probs_softmax.size(1) == 1

        return targets, probs_softmax  # 두가지를 다 보내줘야 추론에 반영 가능

    def reset(self, sequence_length: int) -> None:
        self.memories = self._initialize_memories(
            batch_size=self.batch_size,
            rules=self.rules,
            tokens=self.tokens,
            sequence_length=sequence_length,
        )
        self.sequence_length = sequence_length

    def _initialize_memories(
        self, batch_size: int, rules: dict, tokens: list, sequence_length: int
    ):
        memories = [
            MemoryNode(
                id=idx, rules=rules, tokens=tokens, sequence_length=sequence_length
            )
            for idx, _ in enumerate(range(batch_size))
        ]
        return memories

    def _update(self, targets: torch.Tensor) -> None:
        """각 MemoryNode에 대해

        Args:
            targets (torch.Tensor): [description]
        """
        for t, node in zip(targets, self.memories):
            node.record(t.item())

    @staticmethod
    def _mask(node, vocab_size: int) -> torch.Tensor:
        """MemoryNode의 blacklist 정보를 바탕으로 마스킹 텐서를 생성하는 함수
        마스킹 텐서는 다음 스텝에 활용되어서는 안되는 토큰을 걸러내기 위해 사용

        Returns:
            torch.Tensor: 마스킹 텐서
        """
        blacklist = torch.tensor(node.blacklist)
        output = torch.zeros(vocab_size)
        output = output.scatter(dim=0, index=blacklist, value=1)
        output = output.bool()
        return output


class MemoryNode:
    """
    DecodingManager 내부의 메모리 노드로 활용되는 클래스로, 데이터 샘플별 모델의 토큰 생성 결과 추적에 사용
    DecodingManager가 배치 단위로 데이터를 입력받을 경우, 각 데이터 샘플에 대해 MemoryNode를 활성화,
    추론을 마칠 때까지 스텝별 생성 결과를 추적하며 LaTex 문법에 어긋난 심볼 생성을 방지함
    
    """
    def __init__(self, id: int, rules: dict, tokens: list, sequence_length: int):
        self.id = id  # 배치 내 샘플 순서(0~) NOTE: just for debugging
        self.rules = rules
        self.history = []  # 또는 텐서 - 현재까지 생성된 토큰 리스트 NOTE: just for debugging
        self.tokens = tokens
        self.token2id = {t: i for i, t in enumerate(tokens)}
        self.id2token = {i: t for i, t in enumerate(tokens)}
        self.current_token_id = self._encode("<SOS>")  # 직전 토큰이 무엇인지
        self.num_series = 1  # 같은 토큰이 몇 회 연속으로 등장하고 있는지

        self.cumul_lbrackets = 0
        self.cumul_rbrackets = 0
        # self.residual_length = sequence_length # NOTE: 성능 저하를 유발해 사용하지 않음
        # self.lbrackets_limit = self._get_lbrackets_limit(
        #     self.residual_length, self.cumul_lbrackets
        # )  # 최대 생성 가능한 좌괄호 수

        # 다음 step때 생성하면 안되는 토큰 확인
        self.blacklist = self._look_back()

    def record(self, target_id: int):
        self.history.append(target_id)  # 아직 복잡한 로직이 없어서 필요하지는 않음

        # 연속 등장 카운트
        if self.current_token_id == target_id:
            self.num_series += 1  # 연속 등장
        else:
            self.num_series = 1  # 새 토큰 등장

        if target_id == self._encode("{"):
            self.cumul_lbrackets += 1

        elif target_id == self._encode("}"):
            self.cumul_rbrackets += 1

        self.current_token_id = target_id  # 타깃ID 갱신
        # self.residual_length -= 1
        # self.lbrackets_limit = self._get_lbrackets_limit(
        #     self.residual_length, self.cumul_lbrackets
        # )
        self.blacklist = self._look_back()  # 블랙리스트 갱신

    def _look_back(self) -> list:
        current_token = self._decode(self.current_token_id)
        blacklist = [
            self._encode("<SOS>"),
            self._encode(""),
        ]  # 다음 step에서 생성을 금지할 토큰

        # 괄호 수 균형
        if self.cumul_lbrackets == self.cumul_rbrackets:
            blacklist.append(self._encode("}"))

        # 좌괄호 수 제한
        # if self.lbrackets_limit == 0:
        #     blacklist.append(self._encode("{"))

        # =====CHECK #1 - 첫 step에서 등장할 수 없는 토큰=====
        if current_token == "<EOS>":
            blacklist = sorted(list(set(blacklist)))
            return blacklist

        elif current_token == "<SOS>":
            excepts = [self._encode(t) for t in self.rules["cannot_initial"]]
            blacklist.extend(deepcopy(excepts))
            blacklist = sorted(list(set(blacklist)))
            return blacklist

        # =====CHECK #2 - 다음 target id를 확정지을 수 있는지 여부=====
        # 매우 높은 확률로 뒤에 '_' 토큰이 오는 토큰
        elif current_token in self.rules["next_underbar"]:
            excepts = [
                self._encode(t) for t in self.tokens if t != "_"
            ]  # '_' 외 모든 토큰 블랙
            blacklist.extend(deepcopy(excepts))
            blacklist = sorted(list(set(blacklist)))
            return blacklist

        # 매우 높은 확률로 뒤에 '{' 토큰이 오는 토큰
        elif current_token in self.rules["next_lbracket"]:
            excepts = [
                self._encode(t) for t in self.tokens if t != "{"
            ]  # '{' 외 모든 토큰 블랙
            blacklist.extend(deepcopy(excepts))
            blacklist = sorted(list(set(blacklist)))
            return blacklist

        else:
            # =====CHECK #3 - 다음 target id를 확정지을 수 있는지 여부=====
            # 매우 높은 확률로 뒤에 '_' 토큰이 붙지 않아야 하는 토큰
            if current_token in self.rules["cannot_next_underbar"]:
                blacklist.append(self._encode("_"))  # '_' 추가

            # 매우 높은 확률로 뒤에 '{' 토큰이 붙지 않아야 하는 토큰
            if current_token in self.rules["cannot_next_lbracket"]:
                blacklist.append(self._encode("{"))  # '_' 추가

            # =====CHECK #4 - 최대 연속 생성 횟수 확인=====
            if self.rules["limit_series"][current_token]:
                limit = self.rules["limit_params"][current_token]
                if self.num_series >= limit:  # 연속 횟수가 넘쳤을 경우 블랙
                    blacklist.append(self._encode(current_token))

        blacklist = sorted(list(set(blacklist)))
        return blacklist

    @staticmethod
    def _get_lbrackets_limit(residual_length: int, cumul_lbrackets: int):
        """생성 가능한 좌괄호('{') 개수를 계산하는 함수
        (잔여 좌괄호 생성 가능 횟수) = (남은 생성 횟수) - (현재 좌괄호 생성 횟수) // 2
        """
        limit = (residual_length - cumul_lbrackets) // 2
        if limit <= 0:
            return 0
        return limit

    def _encode(self, token: str) -> int:
        return self.token2id[token]

    def _decode(self, id: int) -> str:
        return self.id2token[id]