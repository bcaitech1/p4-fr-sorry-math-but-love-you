from queue import PriorityQueue

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


def beam_search_ens(models: list, num_steps: int=230):
    
    batch
    for idx in range(num_steps):
        logit_list = []
        for model in models:
            model.cuda()

            dict = model.beam_search(asdfasdfas, step=idx)

            hidden_state = dict.get('hidden_state', None)

            if hidden_state is None: # SATRN


            
            Attention hidden_state - LSTM
            SATRN - None
            logit_list.append(logit)
            model.cpu()
        
        logit = mean()

        asdfasdfasdfasdf
        asd
        Falsesdf
        as
        dfa
        sdf

        asd
        filterasdf
        asdf
        a
        sdf
        asdf

    outut
    return output