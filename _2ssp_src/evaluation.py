"""2SSP evaluation - evaluate_perplexity only (no lm_eval dependency)"""
import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate_perplexity(
    model, input_ids, seq_len=2048, batch_size=1, enable_tqdm=True, device="cuda"
):
    """Eval perplexity for attention pruning (2SSP stage 2)."""
    num_samples = input_ids.numel() // seq_len
    data = []
    for i in range(num_samples):
        data.append(input_ids[:, i * seq_len : (i + 1) * seq_len])

    nll_running = 0
    tokens_processed = 0

    ppl_range = (
        tqdm(range(0, num_samples, batch_size), desc="Calculating perplexity")
        if enable_tqdm
        else range(0, num_samples, batch_size)
    )

    for i in ppl_range:
        j = min(i + batch_size, num_samples)
        inputs = torch.cat(data[i:j]).to(device)
        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
        )
        a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
        b = tokens_processed / (tokens_processed + shift_labels.numel())
        nll_running = a * loss + b * nll_running
        tokens_processed += shift_labels.numel()

    return nll_running.exp().item()
