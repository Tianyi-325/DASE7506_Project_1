"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--save-every', type=int, default=0,
                   help='Save intermediate inference checkpoints at this step interval.')
    p.add_argument('--schedule', choices=['cosine', 'wsd'], default='cosine',
                   help='Learning-rate plan; wsd holds peak LR before a final decay.')
    p.add_argument('--stable-fraction', type=float, default=0.75,
                   help='Fraction of steps at peak LR before WSD decay.')
    p.add_argument('--sampler', choices=['random', 'coverage'], default='random',
                   help='Training-window sampling; coverage reshuffles nonoverlapping windows each pass.')
    p.add_argument('--teacher', type=Path,
                   help='Optional train-only teacher checkpoint for soft-target distillation.')
    p.add_argument('--init-checkpoint', type=Path,
                   help='Optional self-trained checkpoint to continue from with a fresh optimizer.')
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--lr-floor', type=float, default=.1,
                   help='Final multiplier on the peak learning rate.')
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--distill-weight', type=float, default=0.5,
                   help='Weight on teacher soft-target cross entropy, when --teacher is used.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.save_every < 0:
        p.error('--save-every must be nonnegative.')
    if not 0 < args.stable_fraction < 1:
        p.error('--stable-fraction must lie strictly between zero and one.')
    if args.learning_rate <= 0 or not 0 < args.lr_floor <= 1 or args.warmup_steps < 0:
        p.error('Invalid learning-rate or warmup settings.')
    if args.teacher and not 0 <= args.distill_weight <= 1:
        p.error('--distill-weight must lie in [0, 1].')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    init_checkpoint = None
    init_sha = None
    if args.init_checkpoint:
        init_checkpoint = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        if (init_checkpoint['protocol'] != PROTOCOL or
                init_checkpoint['implementation'] != args.implementation or
                init_checkpoint['config'] != config):
            raise ValueError('Initialization checkpoint is incompatible with this model.')
        model.load_state_dict(init_checkpoint['model'])
        init_sha = sha(args.init_checkpoint)
    teacher = None
    teacher_checkpoint = None
    teacher_sha = None
    if args.teacher:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
        teacher_checkpoint = torch.load(args.teacher, map_location='cpu', weights_only=True)
        if teacher_checkpoint['protocol'] != PROTOCOL:
            raise ValueError('Teacher checkpoint uses an incompatible protocol.')
        teacher, _ = make_model(teacher_checkpoint['implementation'],
                                teacher_checkpoint['config'], device)
        teacher.load_state_dict(teacher_checkpoint['model'])
        teacher.eval()
        teacher.requires_grad_(False)
        teacher_sha = sha(args.teacher)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    epoch_starts = torch.empty(0, dtype=torch.long)
    epoch_cursor = 0
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    for step in range(args.steps):
        if args.sampler == 'random':
            starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        else:
            if epoch_cursor + args.batch_size > len(epoch_starts):
                offset = int(torch.randint(256, (1,), generator=rng).item())
                epoch_starts = torch.arange(offset, len(tokens)-257, 256)
                epoch_starts = epoch_starts[torch.randperm(len(epoch_starts), generator=rng)]
                epoch_cursor = 0
            starts = epoch_starts[epoch_cursor:epoch_cursor+args.batch_size].to(device)
            epoch_cursor += args.batch_size
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        if args.schedule == 'cosine':
            decay_progress = step / args.steps
        else:
            stable_steps = int(args.steps * args.stable_fraction)
            decay_progress = max(0., (step - stable_steps) / (args.steps - stable_steps))
        warmup = min(1., (step+1)/args.warmup_steps) if args.warmup_steps else 1.
        learning_rate = args.learning_rate * warmup * (
            args.lr_floor + (1-args.lr_floor)*.5*(1+math.cos(math.pi*decay_progress)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits = model(batch[:,:-1]).float()
            loss = F.cross_entropy(logits.flatten(0,1),batch[:,1:].flatten())
            if teacher is not None:
                with torch.no_grad():
                    teacher_prob = teacher.predict_log_probs(batch[:,:-1]).float().exp()
                soft_loss = -(teacher_prob * F.log_softmax(logits, dim=-1)).sum(-1).mean()
                loss = (1 - args.distill_weight) * loss + args.distill_weight * soft_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
        if args.save_every > 0 and (step+1)%args.save_every == 0 and step+1 < args.steps:
            snapshot = args.run_dir / f'checkpoint_step_{step+1}.pt'
            torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                        'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'seed':args.seed,
                        'train_tokens':((step+1)*args.batch_size*256 +
                                        (init_checkpoint['train_tokens'] if init_checkpoint else 0) +
                                        (teacher_checkpoint['train_tokens'] if teacher_checkpoint else 0)),
                        'training_step':step+1,'training_steps_planned':args.steps,
                        'schedule':args.schedule,'stable_fraction':args.stable_fraction,
                        'initial_checkpoint_sha256':init_sha,
                        'learning_rate':args.learning_rate,'lr_floor':args.lr_floor},snapshot)
            print(json.dumps({'snapshot':str(snapshot),'sha256':sha(snapshot)}),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    own_train_tokens = args.steps*args.batch_size*256
    total_train_tokens = (own_train_tokens +
                          (teacher_checkpoint['train_tokens'] if teacher_checkpoint else 0) +
                          (init_checkpoint['train_tokens'] if init_checkpoint else 0))
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':total_train_tokens,
                'teacher_checkpoint_sha256':teacher_sha,
                'initial_checkpoint_sha256':init_sha,
                'schedule':args.schedule,'stable_fraction':args.stable_fraction,
                'learning_rate':args.learning_rate,'lr_floor':args.lr_floor},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'sampler':args.sampler,
              'schedule':args.schedule,'stable_fraction':args.stable_fraction,
              'learning_rate':args.learning_rate,'lr_floor':args.lr_floor,
              'warmup_steps':args.warmup_steps,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':total_train_tokens,'own_train_tokens':own_train_tokens,
              'teacher_checkpoint_sha256':teacher_sha,'distill_weight':args.distill_weight if teacher is not None else None,
              'initial_checkpoint_sha256':init_sha,
              'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
