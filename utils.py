from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

def get_output(output, config, copy_vocab, tokenzier):
    _BOUNDARY = tokenzier.eos_token_id
    N, D = output.size()
    output = output.detach().cpu()

    out = []
    for i in range(N):
        txt = []
        for j in range(D):
            ix = output[i, j].item()
            if ix == _BOUNDARY: break
            if ix < config.vocab_size:
                txt.append(ix)
            else:
                ix = ix - config.vocab_size
                txt += copy_vocab.token_fg_w[ix]
        out.append(txt)
    return out


class AdamWOpt(object):
    def __init__(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

def build_optimizer(opt, model):
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
	    {
	        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
	        "weight_decay": opt.weight_decay,
	    },
	    {
	        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
	        "weight_decay": 0.0,
	    },
	]
	assert opt.num_training_steps > 0
	optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate, eps=opt.adam_epsilon)
	scheduler = get_constant_schedule(optimizer) #get_linear_schedule_with_warmup(optimizer, opt.warmup_step, opt.num_training_steps, -1) 

	return AdamWOpt(optimizer, scheduler)
