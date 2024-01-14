from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModel
import torch
import argparse
from datasets import load_dataset
from utils import argmax_sampling, fix_state, identity_sampling
from tqdm import tqdm
import time
import json


class NormalEncodingEncDec:
    def __init__(self, model, bos, eos, sampling_stategy=argmax_sampling, device=torch.device('cpu')):
        self.model = model
        self.sampling_stategy = sampling_stategy
        self.device = device
        self.bos = bos.to(device)
        self.eos = eos.to(device)
        model.eval()

    def inference(self, prefix, max_new_tokens, temperature=1.0):
        prefix = prefix.to(self.device)

        with torch.no_grad():
            generated = []
            new_word = self.bos.reshape(1)
            
            model_output = self.model.encoder(input_ids=prefix, return_dict=True)
            last_model_state = None
            model_encoder_outputs = (model_output.last_hidden_state, )
        
            while len(generated) < max_new_tokens and (len(generated) == 0 or generated[-1] != self.eos):
                assert(new_word.shape == (1,))
                output = self.model(None,
                                    encoder_outputs=model_encoder_outputs,
                                    decoder_input_ids=new_word.unsqueeze(0), 
                                    past_key_values=last_model_state,
                                    use_cache=True)
                last_model_state = output.past_key_values
                distribution = output.logits[0, -1, :] / temperature
                assert(len(distribution.shape) == 1)
                distribution = self.sampling_stategy(distribution)          
                probs = torch.nn.functional.softmax(distribution, dim=0)
                idx = torch.searchsorted(probs.cumsum(0), torch.rand(1, device=self.device))
                generated.append(idx.item())
                new_word = idx.reshape(-1)
            return generated, len(generated), len(generated)


class SpecInferencerEncDec:
    def __init__(self, model, draft_model, bos, eos, gamma=5, sampling_stategy=argmax_sampling,
                device=torch.device('cpu')):
        self.gamma = gamma
        self.model = model
        self.draft_model = draft_model
        self.sampling_stategy = sampling_stategy
        self.device = device
        self.bos = bos.to(device)
        self.eos = eos.to(device)
        model.eval()
        dmodel.eval()
    
    def inference(self, prefix, max_new_tokens=32, temperature=1.0):
        prefix = prefix.to(self.device)
        predicted_ok = 0
        iterations = 0

        with torch.no_grad():
            generated = []
            new_word = self.bos.reshape(1)
            
            model_output = self.model.encoder(input_ids=prefix, return_dict=True)
            last_model_state = None
            model_encoder_outputs = (model_output.last_hidden_state, )
            dmodel_output = self.draft_model.encoder(input_ids=prefix, return_dict=True)
            last_dmodel_state = None
            dmodel_encoder_outputs = (dmodel_output.last_hidden_state, )
        
            while len(generated) < max_new_tokens:
                iterations += 1
                proposals = []
                current_new_word = new_word
                for i in range(min(self.gamma, max_new_tokens - len(generated))):
                    assert(current_new_word.shape == (1,))
                    output = self.draft_model(None,
                                            encoder_outputs=dmodel_encoder_outputs,
                                            decoder_input_ids=current_new_word.unsqueeze(0), 
                                            past_key_values=last_dmodel_state,
                                            use_cache=True)
                    last_dmodel_state = output.past_key_values
                    distribution = output.logits[0, -1, :] / temperature
                    assert(len(distribution.shape) == 1)
                    distribution = self.sampling_stategy(distribution)          
                    probs = torch.nn.functional.softmax(distribution, dim=0)
                    idx = torch.searchsorted(probs.cumsum(0), torch.rand(1, device=self.device))
                    proposals.append([idx, probs])
                    current_new_word = idx.reshape(1)
        
                new_words = [new_word] + [el[0] for el in proposals[:-1]]
                output = self.model(None,
                                    encoder_outputs=model_encoder_outputs,
                                    decoder_input_ids=torch.cat(new_words, dim=0).unsqueeze(0), 
                                    past_key_values=last_model_state,
                                    use_cache=True)
                new_words.append(proposals[-1][0])
                last_model_state = output.past_key_values
                logits = output.logits
                true_distributions = []
                for i in range(logits.shape[1]):
                    tmp = self.sampling_stategy(logits[0, i, :]) / temperature
                    assert(len(tmp.shape) == 1)
                    distribution = torch.nn.functional.softmax(tmp, dim=0)
                    true_distributions.append(distribution)
                good_tokens = 1
                
                for i in range(len(true_distributions)):
                    token_idx = proposals[i][0]
                    predicted_ok += 1
                    if proposals[i][1][token_idx] <= true_distributions[i][token_idx]:
                        good_tokens += 1
                    else:
                        rejection_p = 1 - true_distributions[i][token_idx] / proposals[i][1][token_idx]
                        if torch.rand(1, device=device) <= rejection_p:
                            distribution = torch.nn.functional.relu(true_distributions[i] - proposals[i][1])
                            distribution /= distribution.sum()
                            idx = torch.searchsorted(distribution.cumsum(0), torch.rand(1, device=device))
                            new_words[good_tokens] = new_word = idx
                            break
                        else:
                            good_tokens += 1
                    new_word = token_idx
                
                good_tokens = min(good_tokens, len(new_words) - 1)
                for i, el in enumerate(new_words[:good_tokens]):
                    generated.append(el.item())
                    if el == self.eos:
                        return generated, predicted_ok, iterations
                
                rollback = len(new_words) - good_tokens - 1
                last_model_state = fix_state(last_model_state, rollback=rollback)
                last_dmodel_state = fix_state(last_dmodel_state, rollback=rollback)
                new_word = new_word.reshape(1)
                if new_word == self.eos:
                    generated.append(new_word.item())
                    break
                assert(isinstance(new_word, torch.Tensor))
            return generated, predicted_ok, iterations


name2sampling_method = {
    "argmax": argmax_sampling,
    "identity": identity_sampling
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("T5 simple example")
    parser.add_argument("--t", default=1.0, type=float, help="temperature")
    parser.add_argument("--gamma", default=5, type=int, help="how many tokens to generate with draft model")
    parser.add_argument("--model", default="t5-3b", help="hf name of big model")
    parser.add_argument("--draft", default="t5-small", help="hf name of draft model")
    parser.add_argument("--sampling", default="argmax", help="sampling method")
    parser.add_argument("--max_new_tokens", default=64, type=int)
    parser.add_argument("--logfile", default="time.log", help="Output time with configuration and inference time information")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=512)

    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    if args.draft == 'unigram':
        from unigram_utils_t5 import UnigramModel
        dmodel = UnigramModel(32128)
        dmodel.load_state_dict(torch.load('unigram_model.pt'))
        dmodel = dmodel.to(device)
    else:
        dmodel = T5ForConditionalGeneration.from_pretrained(args.draft).to(device)

    input = tokenizer('translate from English to German: I like apple pies', return_tensors='pt')['input_ids']
    bos = torch.tensor(tokenizer.pad_token_id)
    eos = torch.tensor(tokenizer.eos_token_id)
    method = name2sampling_method[args.sampling]

    dataset = load_dataset("wmt14", 'de-en', split="validation")
    prefix = "translate English to German: "

    if args.gamma != 0:
        inferencer = SpecInferencerEncDec(model, dmodel, bos, eos, args.gamma, method, device)
        desc = "Speculative Decoding"
    else:
        inferencer = NormalEncodingEncDec(model, bos, eos, method, device)
        desc = "Normal Decoding"
    
    start = time.time()
    total_iterations = 0
    total_accepted_predictions = 0

    for i, el in enumerate(tqdm(dataset, desc=desc)):
        text = prefix + el['translation']['en']
        prompt = tokenizer(text, return_tensors='pt')['input_ids']
        output, accepted_predictions, iterations = inferencer.inference(prompt, 
                                                                        max_new_tokens=args.max_new_tokens, 
                                                                        temperature=args.t)
        total_iterations += iterations
        total_accepted_predictions += accepted_predictions
    end = time.time()
    elapsed = end - start
    with open(args.logfile, 'a') as file:
        tmp = args.__dict__
        tmp['elapsed'] = elapsed
        tmp['E_tok'] = total_accepted_predictions / total_iterations
        print(json.dumps(tmp), file=file)

