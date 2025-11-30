"""
WiggleGPT v3 - chat.py
======================

Interactive chat with fine-tuned WiggleGPT model.
Uses the same chat template as fine-tuning.

Usage:
    python chat.py
    python chat.py --out_dir=out-wigglegpt-finetune-native
"""
import os
from contextlib import nullcontext
import torch
import tiktoken
from model_bio import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration
out_dir = 'out-wigglegpt-finetune-native'  # Fine-tuned model directory
device = 'cuda'
dtype = 'float16'
temperature = 0.7
top_k = 50
max_new_tokens = 512
repetition_penalty = 1.15
seed = None  # Set to int for reproducible outputs

exec(open('configurator.py', encoding='utf-8').read())
# -----------------------------------------------------------------------------

# Chat template (same as fine-tuning)
USER_START = "<|user|>\n"
USER_END = "\n<|/user|>\n"
ASSISTANT_START = "<|assistant|>\n"
ASSISTANT_END = "\n<|/assistant|>"

def load_model():
    """Load the fine-tuned model"""
    print(f"\nLoading model from {out_dir}...")
    
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # Print info
    if gptconf.use_bio_mlp:
        print("Bio-inspired neurons: ENABLED (oscillating activations)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if 'best_val_loss' in checkpoint:
        print(f"Best val loss: {checkpoint['best_val_loss']:.4f}")
    if 'iter_num' in checkpoint:
        print(f"Trained for: {checkpoint['iter_num']} iterations")
    
    return model, gptconf


def generate_response(model, config, enc, prompt, conversation_history=""):
    """Generate a response to the user prompt"""
    
    # Build the full prompt with history
    full_prompt = conversation_history
    full_prompt += USER_START + prompt + USER_END
    full_prompt += ASSISTANT_START
    
    # Encode
    input_ids = enc.encode(full_prompt, allowed_special={"<|endoftext|>"})
    
    # Ensure minimum length for RoPE
    min_length = config.n_head
    if len(input_ids) < min_length:
        padding = enc.encode(' ')[0]
        input_ids = [padding] * (min_length - len(input_ids)) + input_ids
    
    # Truncate if too long (keep most recent context)
    max_context = config.block_size - max_new_tokens
    if len(input_ids) > max_context:
        input_ids = input_ids[-max_context:]
    
    x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]
    
    # Setup
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Generate
    generated_tokens = []
    with torch.no_grad():
        with ctx:
            idx = x
            for _ in range(max_new_tokens):
                # Crop if needed
                idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size:]
                
                # Forward
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(idx[0].tolist()[-100:]):  # Last 100 tokens
                        logits[0, token_id] /= repetition_penalty
                
                # Top-k
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                
                # Decode new token
                new_token = idx_next[0, 0].item()
                generated_tokens.append(new_token)
                
                # Check for end of response
                decoded_so_far = enc.decode(generated_tokens)
                if "<|/assistant|>" in decoded_so_far or "<|endoftext|>" in decoded_so_far:
                    break
                if "<|user|>" in decoded_so_far:  # Model trying to simulate user
                    break
    
    # Decode response
    response = enc.decode(generated_tokens)
    
    # Clean up
    response = response.split("<|/assistant|>")[0]
    response = response.split("<|user|>")[0]
    response = response.split("<|endoftext|>")[0]
    response = response.strip()
    
    return response


def main():
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Load model
    model, config = load_model()
    enc = tiktoken.get_encoding("gpt2")
    
    print("\n" + "=" * 60)
    print("WIGGLEGPT CHAT")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end the conversation.")
    print("Type 'clear' to reset conversation history.")
    print("=" * 60 + "\n")
    
    conversation_history = ""
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            conversation_history = ""
            print("[Conversation cleared]\n")
            continue
        
        # Generate response
        response = generate_response(model, config, enc, user_input, conversation_history)
        
        # Update history
        conversation_history += USER_START + user_input + USER_END
        conversation_history += ASSISTANT_START + response + ASSISTANT_END + "\n"
        
        print(f"\nWiggleGPT: {response}\n")


if __name__ == '__main__':
    main()
