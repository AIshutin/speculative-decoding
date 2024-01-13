import torch


def argmax_sampling(logits): # Tensor [vocab_size]
    assert(len(logits.shape) == 1)
    argmax_id = logits.max(dim=-1).indices
    logits = -100 * torch.ones_like(logits)
    logits[argmax_id] = 100
    return logits


def fix_state(state, rollback):
    if rollback == 0:
        return state
    state = list(state)
    for i in range(len(state)):
        assert(isinstance(state[i], tuple))
        state[i] = list(state[i])
        for j in range(len(state[i])):
            state[i][j] = state[i][j][..., :-rollback, :]
    return state