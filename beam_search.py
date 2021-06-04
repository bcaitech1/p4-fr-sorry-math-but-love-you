# from queue import PriorityQueue

# class BeamSearchNode(object):
#     def __init__(self, hidden_state, prev_node, token_id, log_prob, length):
#         self.hidden_state = hidden_state
#         self.prev_node = prev_node
#         self.token_id = token_id
#         self.logp = log_prob
#         self.len = length  # 길이

#     def eval(self, alpha=0.75) -> float:
#         """score 측정 함수
#         score: L - 해당 스텝까지의 총 길이, alpha - 페널티항, c: context
#             {1/L^{alpha}}*log{P(y_{1}, ..., y_{L}|c)}
#             = {1/L^{alpha}}*SUM_{t'=1}^{L}{log{P(y_{1}, ..., y_{L}|c)}}
#         Args:
#             alpha (float, optional): 스텝 길이에 따른 페널티를 위한 파라미터. Defaults to 0.75.

#         Returns:
#             score (float): 점수
        
#         References:
#             - Beam Search, Dive into Deep Learning,
#               https://d2l.ai/chapter_recurrent-modern/beam-search.html#id1
#         """
#         # reward = 0
#         return self.logp / (float(self.len)**alpha)
#         # return self.logp / float(self.len - 1 + 1e-6) + alpha * reward

#     def __lt__(self, other):
#         return self.len < other.len

#     def __gt__(self, other):
#         return self.len > other.len