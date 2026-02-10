"""
CFSP-style block (layer) influence: 입력-출력 거리로 레이어 민감도 계산
- 레이어 입력과 출력 간 distance를 구해, 그 증분을 민감도(중요도)로 사용
"""
import torch


def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    metrics: str = "angular",
):
    """
    레이어 입력/출력 간 거리(민감도) 계산.
    input_hidden_state: (B, S, D)
    output_hidden_state: (B, S, D)
    metrics: 'angular' | 'cosine' | 'mse' | 'mae'
    반환: (B*S,) 또는 스칼라에 가까운 텐서 (평균 등)
    """
    _, _, d = input_hidden_state.shape
    inp = input_hidden_state.reshape(-1, d).float()
    out = output_hidden_state.reshape(-1, d).float()

    if metrics == "mse":
        return torch.mean((inp - out) ** 2, dim=-1)
    if metrics == "mae":
        return torch.mean(torch.abs(inp - out), dim=-1)

    norm_inp = inp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    norm_out = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = (inp * out).sum(dim=-1) / (norm_inp.squeeze(-1) * norm_out.squeeze(-1) + 1e-8)
    sim = sim.clamp(-1.0, 1.0)

    if metrics == "angular":
        return (torch.arccos(sim) / torch.pi).nan_to_num(nan=0.5)
    return (1.0 - sim).nan_to_num(nan=0.5)
