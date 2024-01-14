import torch


class EncoderOutput:
    last_hidden_state = None


class ModelOutput:
    past_key_values = None
    logits = None


class UnigramModel(torch.nn.Module):
    def __init__(self, vocab_size=32100) -> None:
        super().__init__()
        self.cooc = torch.nn.Parameter(torch.zeros(vocab_size, vocab_size, dtype=torch.float))
        self.encoder = lambda *args, **kwargs: EncoderOutput()
    
    def forward(self, _, decoder_input_ids, **kwargs):
        assert(decoder_input_ids.shape[0] == 1)
        last_token = decoder_input_ids[0, -1]
        out = ModelOutput()
        out.logits = self.cooc[last_token].reshape(1, 1, -1)
        return out


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from tqdm import tqdm
    tokenizer = AutoTokenizer.from_pretrained("t5-3b")
    dataset = load_dataset("wmt14", 'de-en', split="train")
    vocab_size = 32128
    print('Vocab size:', vocab_size)
    
    matrix = torch.zeros(vocab_size, vocab_size, dtype=torch.float)
    for i, el in enumerate(tqdm(dataset)):
        ids = tokenizer(el['translation']['de'], return_tensors='pt')['input_ids'][0]
        for prev_id, next_id in zip(ids[:-1], ids[1:]):
            matrix[prev_id][next_id] += 1
    matrix = torch.log(matrix / (matrix.sum(dim=-1, keepdim=True) + 1e-8) + 1e-8)
    model = UnigramModel(vocab_size)
    model.cooc = torch.nn.Parameter(matrix)
    torch.save(model.state_dict(), 'unigram_model.pt')
