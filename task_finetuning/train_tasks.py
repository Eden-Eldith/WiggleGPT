"""Full-model GPT-1-style task fine-tuning of original WiggleGPT.

Native model_bio.py is imported unchanged. A final-state classifier is added;
its language-model objective uses the native tied projection. No CPU forwards.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT / 'native'
sys.path.insert(0, str(NATIVE))
from task_runtime import load_model
import model_bio
import torch
from torch import nn
from torch.nn import functional as F

STOP = False
MARKERS = {'start': 50257, 'delimiter': 50258, 'classify': 50259, 'pad': 50256}

def stop_requested(signum, frame):
    global STOP
    STOP = True
    print('Stop requested; finishing the update and saving latest.pt.', flush=True)

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

class TaskModel(nn.Module):
    def __init__(self, base, task, dropout):
        super().__init__()
        self.base = base
        self.task = task
        self.dropout = dropout
        # Added linear classifier is the requested GPT-1 task head, not a backbone replacement.
        with torch.device('cuda:0'):
            self.classifier = nn.Linear(base.config.n_embd, 2 if task == 'sst2' else 1)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        self.hidden = None
        self.base.transformer.ln_f.register_forward_hook(self.capture_hidden)

    def capture_hidden(self, module, args, output):
        self.hidden = output

    def forward(self, tokens, lengths, auxiliary=False):
        batch, choices, width = tokens.shape
        flat = tokens.reshape(batch * choices, width)
        self.base(flat)
        hidden = self.hidden
        self.hidden = None
        positions = lengths.reshape(-1) - 1
        features = hidden[torch.arange(batch * choices, device=flat.device), positions]
        features = features.reshape(batch, choices, -1)
        # GPT-1's multiple-choice head shares a dropout feature mask across options.
        if self.training and self.dropout:
            mask = F.dropout(torch.ones((batch, 1, features.shape[-1]), device=features.device,
                                        dtype=features.dtype), p=self.dropout, training=True)
            features = features * mask
        scores = self.classifier(features).squeeze(1) if self.task == 'sst2' else self.classifier(features).squeeze(-1)
        lm_loss = None
        if auxiliary:
            logits = self.base.lm_head(hidden[:, :-1])
            targets = flat[:, 1:].long()
            per_token = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                        targets.reshape(-1), reduction='none').view_as(targets)
            mask = torch.arange(width - 1, device=flat.device)[None, :] < (lengths.reshape(-1, 1) - 1)
            # Average within each candidate sequence, then across all candidates.
            lm_loss = ((per_token * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
        return scores.float(), lm_loss

def load_data(task, device):
    prefix = 'sst2' if task == 'sst2' else 'race_middle'
    splits = ['train', 'validation'] + (['test'] if task != 'sst2' else [])
    data = {}
    for split in splits:
        path = ROOT / 'data' / f'{prefix}_{split}_encoded.pt'
        if not path.exists():
            # Preparation script may use the compact .pt suffix.
            path = ROOT / 'data' / f'{prefix}_{split}.pt'
        cache = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        # Both conventional names are accepted; contents stay unchanged.
        ids = cache.get('input_ids', cache.get('tokens'))
        data[split] = {'input_ids': ids.to(device), 'lengths': cache['lengths'].to(device),
                       'labels': cache['labels'].to(device), 'cache_path': str(path),
                       'metadata': {k:v for k,v in cache.items() if not torch.is_tensor(v)}}
    return data

def select_batch(data, index):
    return data['input_ids'][index], data['lengths'][index], data['labels'][index]

def evaluate(model, data, batch_size, fixed_indices=None):
    model.eval()
    predictions, losses = [], []
    if fixed_indices is None:
        fixed_indices = torch.arange(data['labels'].shape[0], device='cuda:0')
    # AMP-COMPLEX: native weights/activations are real; native RoPE explicitly casts its complex rotation to float32.
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):  # lint: ok AMP-COMPLEX real native weights; RoPE explicitly uses float32 complex rotation
        for start in range(0, len(fixed_indices), batch_size):
            indices = fixed_indices[start:start + batch_size]
            x, lengths, labels = select_batch(data, indices)
            logits, _ = model(x, lengths)
            predictions.append(logits.argmax(-1))
            losses.append(F.cross_entropy(logits, labels, reduction='sum'))
    predicted = torch.cat(predictions)
    labels = data['labels'][fixed_indices]
    correct = int((predicted == labels).sum())
    counts = torch.bincount(labels, minlength=2 if model.task == 'sst2' else 4).cpu().tolist()
    return {'correct': correct, 'total': len(labels), 'accuracy': correct / len(labels),
            'loss': float(torch.stack(losses).sum() / len(labels)),
            'predictions': predicted.cpu().tolist(), 'gold_label_counts': counts,
            'majority_accuracy': max(counts) / len(labels)}

def optimizer_for(model, config):
    decay, no_decay = [], []
    protected = ('omega', 'phi', 'baseline', 'norm', 'bias', 'scale', 'phase')
    for name, parameter in model.named_parameters():
        if parameter.ndim >= 2 and not parameter.is_complex() and not any(word in name for word in protected):
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    print(f'Decay: {sum(p.numel() for p in decay):,}; protected/no decay: {sum(p.numel() for p in no_decay):,}', flush=True)
    return torch.optim.AdamW([{'params':decay, 'weight_decay':config['weight_decay']},
                              {'params':no_decay, 'weight_decay':0.0}],
        lr=config['learning_rate'], betas=tuple(config['betas']), eps=config['epsilon'], foreach=True)

def schedule(step, total, config):
    warmup = max(1, round(total * config['warmup_fraction']))
    if step < warmup:
        return config['learning_rate'] * (step + 1) / warmup
    return config['learning_rate'] * max(0.0, (total - step) / max(1, total - warmup))

def atomic_save(path, payload):
    temporary = path.with_suffix('.pending.pt')
    torch.save(payload, temporary)
    temporary.replace(path)

def rng_state(generator):
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(0), 'batch_generator': generator.get_state()}

def restore_rng(state, generator):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu(), 0)
    generator.set_state(state['batch_generator'].cpu())

def full_summary(result):
    return {k:v for k,v in result.items() if k != 'predictions'}

def raw_rows(task, split):
    path = ROOT / 'data' / f'{task}_{split}.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

def run_task(task, config, resume=None):
    task_config = config['tasks'][task]
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    base, native_config, encoder, metadata = load_model((ROOT / config['starting_checkpoint']).resolve())
    free, total = torch.cuda.mem_get_info()
    cap = min(int(config['max_process_gib'] * 1024**3),
              total - int(config['desktop_reserve_gib'] * 1024**3),
              free + torch.cuda.memory_reserved() - int(0.5 * 1024**3))
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    base.config.gradient_checkpointing = task_config['gradient_checkpointing']
    metadata['checkpoint_sha256'] = digest((ROOT / config['starting_checkpoint']).resolve())
    model = TaskModel(base, task, config['classifier_dropout'])
    # Reuse three existing padding slots; real GPT-2 IDs and tokenizer are unchanged.
    with torch.no_grad():
        marker_ids = torch.tensor([MARKERS['start'],MARKERS['delimiter'],MARKERS['classify']], device='cuda:0')
        base.transformer.wte.weight.index_copy_(0, marker_ids,
            torch.randn((3, base.config.n_embd), device='cuda:0') * 0.02)
    optimizer = optimizer_for(model, config)
    data = load_data(task, 'cuda:0')
    n_train = data['train']['labels'].shape[0]
    per_epoch = math.ceil(n_train / config['effective_batch_size'])
    total_steps = per_epoch * config['epochs']
    generator = torch.Generator(device='cuda:0').manual_seed(config['seed'] + 123)
    if resume:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False, mmap=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        state = checkpoint['state']
        order = checkpoint['order'].to('cuda:0') if checkpoint['order'] is not None else None
        restore_rng(checkpoint['rng'], generator)
        run_dir = Path(resume).resolve().parent
        del checkpoint
    else:
        run_dir = ROOT / 'runs' / f'train_tasks_{datetime.now(ZoneInfo("Europe/London")).strftime("%Y-%m-%d_%H-%M-%S")}_{task}'
        run_dir.mkdir(parents=True, exist_ok=False)
        state = {'step':0, 'epoch':0, 'offset':0, 'best_accuracy':-1.0, 'best_loss':float('inf'),
                 'best_step':0, 'nonfinite_updates':0, 'consecutive_nonfinite':0, 'last_eval_step':-1,
                 'elapsed_seconds':0.0, 'lr_scale':1.0}
        order = None
        (run_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        shutil.copy2(__file__, run_dir / 'train_tasks.py')
        shutil.copy2(model_bio.__file__, run_dir / 'model_bio.py')
        provenance = {'model_source_sha256':digest(model_bio.__file__), 'trainer_sha256':digest(__file__),
                      'starting_checkpoint':metadata, 'torch':torch.__version__, 'markers':MARKERS,
                      'native_backbone_dropout':base.config.dropout,
                      'data':{split:d['metadata'] for split,d in data.items()},
                      'all_backbone_parameters_train':True}
        (run_dir/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    log = (run_dir / (run_dir.name + '.log')).open('a', encoding='utf-8', buffering=1)
    def emit(event):
        event.update(task=task)
        line = json.dumps(event, ensure_ascii=False)
        print(line, flush=True)
        log.write(line + '\n')
        with (run_dir/'metrics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(line + '\n')
    def save(name):
        atomic_save(run_dir/name, {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                    'state':dict(state), 'order':order, 'rng':rng_state(generator), 'config':config,
                    'task':task, 'model_args':vars(base.config), 'markers':MARKERS,
                    'source_hashes':{'model':digest(model_bio.__file__),'trainer':digest(__file__)}})
    evaluation_batch = 16 if task == 'sst2' else 4
    heldout = 'validation' if task == 'sst2' else 'test'
    before_path = run_dir/'before.json'
    start_time = time.perf_counter()
    elapsed_prior = state['elapsed_seconds']
    emit({'event':'start','step':state['step'],'train_examples':n_train,'epochs':config['epochs'],
          'total_steps':total_steps,'microbatch':task_config['microbatch_size'],
          'effective_batch':config['effective_batch_size'],'memory_cap_gib':round(cap/1024**3,2),
          'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
          'device':torch.cuda.get_device_name(0)})
    if not before_path.exists():
        before = evaluate(model, data[heldout], evaluation_batch)
        before_path.write_text(json.dumps(before,indent=2),encoding='utf-8')
        emit({'event':'before','heldout':heldout,**full_summary(before)})
    else:
        before = json.loads(before_path.read_text(encoding='utf-8'))
    train_fixed = torch.arange(min(128,n_train), device='cuda:0')
    def evaluate_and_save():
        validation = evaluate(model, data['validation'], evaluation_batch)
        train_eval = evaluate(model, data['train'], evaluation_batch, train_fixed)
        changed = (validation['accuracy'] > state['best_accuracy'] or
                   (validation['accuracy'] == state['best_accuracy'] and validation['loss'] < state['best_loss']))
        if changed:
            state['best_accuracy'] = validation['accuracy']
            state['best_loss'] = validation['loss']
            state['best_step'] = state['step']
        state['last_eval_step'] = state['step']
        state['elapsed_seconds'] = elapsed_prior + time.perf_counter() - start_time
        # Once-per-evaluation diagnostic over twelve oscillator tensors, not Python-built training batches.
        omegas = torch.cat([p.detach().reshape(-1) for name,p in model.named_parameters() if name.endswith('.omega')])  # lint: ok STACK-COMPREHENSION evaluation statistics over twelve oscillator tensors
        emit({'event':'eval','step':state['step'],'epoch':state['epoch'],'offset':state['offset'],
              'train':full_summary(train_eval),'validation':full_summary(validation),
              'best_step':state['best_step'],'omega_std':float(omegas.std(correction=0)),
              'peak_allocated_gib':round(torch.cuda.max_memory_allocated()/1024**3,3),
              'sample':{'gold':int(data['validation']['labels'][0]),'predicted':validation['predictions'][0]},
              'elapsed_seconds':round(state['elapsed_seconds'],1)})
        if changed:
            save('best.pt')
        save('latest.pt')
        model.train()
    if state['last_eval_step'] < 0:
        evaluate_and_save()
    model.train()
    accumulated = torch.zeros(3,device='cuda:0')
    logged_steps = 0
    log_start = time.perf_counter()
    stopping_cause = 'epochs_complete'
    while state['epoch'] < config['epochs']:
        if order is None:
            order = torch.randperm(n_train,device='cuda:0',generator=generator)
        while state['offset'] < n_train:
            end = min(n_train,state['offset']+config['effective_batch_size'])
            indices = order[state['offset']:end]
            effective = len(indices)
            lr = schedule(state['step'], total_steps, config) * state['lr_scale']
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            batch_values = torch.zeros(3,device='cuda:0')
            for position in range(0,effective,task_config['microbatch_size']):
                micro = indices[position:position+task_config['microbatch_size']]
                x,lengths,labels = select_batch(data['train'],micro)
                # AMP-COMPLEX: real-valued model; native rotary complex operations remain explicitly float32.
                with torch.autocast('cuda',dtype=torch.bfloat16):  # lint: ok AMP-COMPLEX real native weights; RoPE explicitly uses float32 complex rotation
                    logits,lm_loss = model(x,lengths,auxiliary=True)
                    classification_loss = F.cross_entropy(logits,labels)
                    loss = classification_loss + config['lm_coefficient'] * lm_loss
                    weighted = loss * (len(micro)/effective)
                weighted.backward()
                batch_values += torch.stack((loss.detach(),classification_loss.detach(),lm_loss.detach())) * (len(micro)/effective)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'])
            if torch.isfinite(norm):
                optimizer.step()
                state['consecutive_nonfinite'] = 0
            else:
                state['nonfinite_updates'] += 1
                state['consecutive_nonfinite'] += 1
                if state['consecutive_nonfinite'] >= 2:
                    state['lr_scale'] *= 0.5
                emit({'event':'nonfinite_update_skipped','step':state['step']+1,
                      'count':state['nonfinite_updates'],'lr_scale':state['lr_scale']})
            state['step'] += 1
            state['offset'] = end
            accumulated += batch_values
            logged_steps += 1
            if state['step'] == 1 or state['step'] % config['log_interval'] == 0:
                # One host transfer per logging interval; hot microbatches remain on CUDA.
                values = (accumulated/logged_steps).cpu().tolist()  # lint: ok SYNC-IN-LOOP once per logging interval
                state['elapsed_seconds'] = elapsed_prior + time.perf_counter()-start_time
                emit({'event':'train','step':state['step'],'total_steps':total_steps,
                      'epoch_fraction':round(state['epoch']+end/n_train,4),'loss':values[0],
                      'classification_loss':values[1],'lm_loss':values[2],'lr':lr,
                      'updates_per_second':round(logged_steps/(time.perf_counter()-log_start),3),
                      'peak_allocated_gib':round(torch.cuda.max_memory_allocated()/1024**3,3),
                      'nonfinite_updates':state['nonfinite_updates'],'elapsed_seconds':round(state['elapsed_seconds'],1)})
                accumulated.zero_(); logged_steps=0;log_start=time.perf_counter()
            if state['step'] % task_config['eval_interval'] == 0:
                evaluate_and_save()
                log_start=time.perf_counter()
            elif state['step'] % config['checkpoint_interval'] == 0:
                state['elapsed_seconds'] = elapsed_prior + time.perf_counter()-start_time
                save('latest.pt')
            if STOP or state['consecutive_nonfinite'] >= config['max_consecutive_nonfinite']:
                stopping_cause = 'user_interrupt' if STOP else 'nonfinite_limit'
                state['elapsed_seconds'] = elapsed_prior + time.perf_counter()-start_time
                save('latest.pt')
                emit({'event':'stopped','stopping_cause':stopping_cause,'step':state['step']})
                log.close()
                return False
        state['epoch'] += 1
        state['offset'] = 0
        order = None
        if state['last_eval_step'] != state['step']:
            evaluate_and_save()
        else:
            save('latest.pt')
    # Final held-out scores use the best validation-selected checkpoint.
    best = torch.load(run_dir/'best.pt',map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(best['model'],strict=True)
    best_step = best['state']['step']
    del best
    after = evaluate(model,data[heldout],evaluation_batch)
    (run_dir/'after.json').write_text(json.dumps(after,indent=2),encoding='utf-8')
    rows = raw_rows(task,heldout)
    examples = []
    for row_index,(row,pb,pa) in enumerate(zip(rows,before['predictions'],after['predictions'])):
        example = {'id':f'{task}/{heldout}/{row_index}','input':row.get('text',row.get('article','')),
                   'prompt':row.get('question',''),'options':row.get('options',['negative','positive']),
                   'gold':row['label'],'predicted_before':pb,'predicted_after':pa,
                   'correct_before':pb==row['label'],'correct_after':pa==row['label']}
        examples.append(example)
    (run_dir/'heldout_predictions.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')
    probe_count = 100 if task=='sst2' else 40
    probe = {'before_correct':sum(e['correct_before'] for e in examples[:probe_count]),
             'after_correct':sum(e['correct_after'] for e in examples[:probe_count]),'total':probe_count}
    state['elapsed_seconds'] = elapsed_prior+time.perf_counter()-start_time
    summary = {'id':task,'name':'SST-2 sentiment' if task=='sst2' else 'RACE middle-school reading',
               'train_examples':n_train,'validation_examples':len(data['validation']['labels']),
               'test_examples':len(data['test']['labels']) if task!='sst2' else 1821,
               'heldout_split':heldout,'baseline':{'label':'Before task training (same classifier)',**full_summary(before)},
               'finetuned':full_summary(after),'chance_accuracy':0.5 if task=='sst2' else 0.25,
               'majority_accuracy':after['majority_accuracy'],'paired_earlier_probe':probe,
               'training':{'epochs':config['epochs'],'max_steps':total_steps,'steps':state['step'],
                           'batch_size':config['effective_batch_size'],'microbatch_size':task_config['microbatch_size'],
                           'gradient_accumulation':config['effective_batch_size']//task_config['microbatch_size'],
                           'learning_rate':config['learning_rate'],'max_length':config['max_length'],
                           'lm_coefficient':config['lm_coefficient'],'stopping_cause':stopping_cause,
                           'classifier_dropout':config['classifier_dropout'],'native_backbone_dropout':base.config.dropout,
                           'nonfinite_updates':state['nonfinite_updates'],'best_step':best_step,
                           'peak_allocated_gib':round(torch.cuda.max_memory_allocated()/1024**3,3),
                           'trainable_parameters':sum(p.numel() for p in model.parameters()),
                           'head':'768-to-2 linear' if task=='sst2' else 'shared 768-to-1 linear over four answers',
                           'checkpoint_selection':'highest validation accuracy, lower validation classification loss breaks ties',
                           'details':'Full native backbone trained. Zero decay on oscillator and 1-D parameters. Linear GPT-1 warmup/decay, AdamW, gradient clip1.0. Original backbone dropout retained; classifier dropout0.1. Three unused padding-vocabulary slots initialized for task markers.'},
               'examples':examples,'checkpoint':{'name':str((run_dir/'best.pt').relative_to(ROOT)),
                                                'sha256':digest(run_dir/'best.pt')},
               'elapsed_seconds':state['elapsed_seconds']}
    (run_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    emit({'event':'complete','step':state['step'],'best_step':best_step,'stopping_cause':stopping_cause,
          'heldout':heldout,'before':full_summary(before),'after':full_summary(after),
          'paired_probe':probe,'elapsed_seconds':round(state['elapsed_seconds'],1)})
    log.close()
    del model,base,optimizer,data
    gc.collect();torch.cuda.empty_cache()
    return True

def write_comparison():
    tasks=[]
    for task in ['sst2','race_middle']:
        paths=sorted((ROOT/'runs').glob(f'*_{task}/summary.json'),key=lambda path:path.stat().st_mtime)
        if not paths:
            return
        tasks.append(json.loads(paths[-1].read_text(encoding='utf-8')))
    data={'date':datetime.now(ZoneInfo('Europe/London')).isoformat(),
          'scope':'Full SST-2 training and validation; full RACE-middle training, validation selection and held-out test. Historical GPT-1 figures use different training coverage and test sets.',
          'method':'Two independent full-model adaptations from original pretrained WiggleGPT. Classifier cross-entropy plus0.5 times token-average auxiliary language-model cross-entropy, with GPT-1-style input markers and linear task heads.',
          'control_method':'Identical task inputs and initialized classifier evaluated before task training; after scores use validation-selected trained checkpoint. Earlier language-model likelihood probes are separately identified.',
          'historical':{'gpt1_race_accuracy':59.0,'gpt1_race_middle_accuracy':62.9,'gpt1_sst2_accuracy':91.3,
                         'source_url':'https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf',
                         'note':'GPT-1 SST-2 91.3% is GLUE test, while this run reports public validation. GPT-1 RACE-M62.9% used combined RACE-middle/high training, while Wiggle used middle-only. Historical rows give context and are not a matched-model replication.'},
          'tasks':tasks,'elapsed_seconds':sum(t['elapsed_seconds'] for t in tasks)}
    (ROOT/'comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['all','sst2','race_middle'],default='all')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'train_config.json')
    parser.add_argument('--starting-checkpoint',type=Path,help='Original pretrained checkpoint; overrides config for a new run.')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.starting_checkpoint:
        config['starting_checkpoint']=str(args.starting_checkpoint.expanduser().resolve())
    torch.set_num_threads(config['cpu_threads'])
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
    except (ImportError, OSError):
        print('Optional process priority setting unavailable; continuing with configured CPU threads.',flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU model execution is disabled.')
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')
    signal.signal(signal.SIGINT,stop_requested)
    if args.resume:
        resume_meta=torch.load(args.resume,map_location='cpu',weights_only=False,mmap=True)
        selected=[resume_meta['task']]
        del resume_meta
    else:
        selected=['sst2','race_middle'] if args.task=='all' else [args.task]
    for task in selected:
        if not run_task(task,config,args.resume):
            return
        write_comparison()
    print('All requested task adaptations complete.',flush=True)

if __name__=='__main__':
    main()
