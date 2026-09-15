"""Latent marked-event utilities used by the TracePoint-VAD pilot."""

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TracePointDiagnostics:
    """Aggregate interpretable event-process statistics over an evaluation split."""

    def __init__(self, duration_bins: Sequence[int]):
        if not duration_bins:
            raise ValueError("duration_bins must be non-empty")
        self.duration_bins = tuple(int(duration) for duration in duration_bins)
        self.duration_mass = [0.0 for _ in self.duration_bins]
        self.total_expected_starts = 0.0
        self.positive_video_count = 0
        self.normal_video_count = 0
        self.positive_event_score_sum = 0.0
        self.normal_event_score_sum = 0.0
        self.positive_snippet_count = 0
        self.normal_snippet_count = 0
        self.positive_expected_starts_sum = 0.0
        self.normal_expected_starts_sum = 0.0
        self.positive_hurdle_presence_sum = 0.0
        self.normal_hurdle_presence_sum = 0.0
        self.positive_hurdle_global_presence_sum = 0.0
        self.normal_hurdle_global_presence_sum = 0.0

    def update(
        self,
        event_score: Tensor,
        start_prob: Tensor,
        is_normal: bool,
        hurdle_presence_prob: Optional[Tensor] = None,
        hurdle_global_presence_prob: Optional[Tensor] = None,
    ) -> None:
        if start_prob.ndim != 4 or start_prob.shape[-1] != len(self.duration_bins):
            raise ValueError("start_prob must have shape [batch, time, class, duration]")

        scores = event_score.detach().reshape(-1).float().cpu()
        duration_mass = start_prob.detach().sum(dim=(0, 1, 2)).float().cpu()
        expected_starts = float(duration_mass.sum().item())
        self.duration_mass = [
            current + float(mass.item())
            for current, mass in zip(self.duration_mass, duration_mass)
        ]
        self.total_expected_starts += expected_starts
        hurdle_presence = None
        if hurdle_presence_prob is not None:
            hurdle_presence = float(
                hurdle_presence_prob.detach().float().mean().cpu().item()
            )
        hurdle_global_presence = None
        if hurdle_global_presence_prob is not None:
            hurdle_global_presence = float(
                hurdle_global_presence_prob.detach().float().mean().cpu().item()
            )

        if is_normal:
            self.normal_video_count += 1
            self.normal_event_score_sum += float(scores.sum().item())
            self.normal_snippet_count += int(scores.numel())
            self.normal_expected_starts_sum += expected_starts
            if hurdle_presence is not None:
                self.normal_hurdle_presence_sum += hurdle_presence
            if hurdle_global_presence is not None:
                self.normal_hurdle_global_presence_sum += hurdle_global_presence
        else:
            self.positive_video_count += 1
            self.positive_event_score_sum += float(scores.sum().item())
            self.positive_snippet_count += int(scores.numel())
            self.positive_expected_starts_sum += expected_starts
            if hurdle_presence is not None:
                self.positive_hurdle_presence_sum += hurdle_presence
            if hurdle_global_presence is not None:
                self.positive_hurdle_global_presence_sum += hurdle_global_presence

    @staticmethod
    def _mean(total: float, count: int) -> float | None:
        return total / count if count else None

    def as_dict(self) -> Dict[str, object]:
        duration_total = sum(self.duration_mass)
        distribution = (
            [mass / duration_total for mass in self.duration_mass]
            if duration_total > 0.0
            else [0.0 for _ in self.duration_mass]
        )
        return {
            "total_expected_starts": self.total_expected_starts,
            "positive_video_count": self.positive_video_count,
            "normal_video_count": self.normal_video_count,
            "positive_mean_event_score": self._mean(
                self.positive_event_score_sum, self.positive_snippet_count
            ),
            "normal_mean_event_score": self._mean(
                self.normal_event_score_sum, self.normal_snippet_count
            ),
            "positive_mean_expected_starts": self._mean(
                self.positive_expected_starts_sum, self.positive_video_count
            ),
            "normal_mean_expected_starts": self._mean(
                self.normal_expected_starts_sum, self.normal_video_count
            ),
            "positive_mean_hurdle_presence": self._mean(
                self.positive_hurdle_presence_sum, self.positive_video_count
            ),
            "normal_mean_hurdle_presence": self._mean(
                self.normal_hurdle_presence_sum, self.normal_video_count
            ),
            "positive_mean_hurdle_global_presence": self._mean(
                self.positive_hurdle_global_presence_sum, self.positive_video_count
            ),
            "normal_mean_hurdle_global_presence": self._mean(
                self.normal_hurdle_global_presence_sum, self.normal_video_count
            ),
            "duration_bins": list(self.duration_bins),
            "duration_distribution": distribution,
        }


def valid_mask_from_lengths(
    lengths: Optional[Tensor], max_length: int, device: torch.device
) -> Tensor:
    if lengths is None:
        return torch.ones(1, max_length, dtype=torch.bool, device=device)

    lengths = torch.as_tensor(lengths, device=device).reshape(-1).long()
    lengths = lengths.clamp(min=0, max=max_length)
    steps = torch.arange(max_length, device=device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


def _prepare_spans(
    span_lengths: Optional[Tensor], valid_mask: Tensor, dtype: torch.dtype
) -> Tensor:
    if span_lengths is None:
        return valid_mask.to(dtype)

    spans = torch.as_tensor(
        span_lengths, device=valid_mask.device, dtype=dtype
    )
    if spans.shape != valid_mask.shape:
        raise ValueError(
            "span_lengths must have shape [batch, time], got "
            f"{tuple(spans.shape)} for {tuple(valid_mask.shape)}"
        )

    spans = torch.where(
        valid_mask & (spans > 0), spans, torch.ones_like(spans)
    )
    return spans * valid_mask.to(dtype)


def project_event_coverage(
    start_prob: Tensor,
    duration_bins: Sequence[int],
    lengths: Optional[Tensor] = None,
    span_lengths: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Project latent start/duration probabilities into snippet coverage.

    ``duration_bins`` are measured in median valid input spans.  When a training
    sample was uniformly compressed, ``span_lengths`` retains how many original
    feature intervals each input token represents.
    """

    if start_prob.ndim != 4:
        raise ValueError("start_prob must have shape [batch, time, class, duration]")

    batch, time, _, duration_count = start_prob.shape
    if len(duration_bins) != duration_count:
        raise ValueError("duration_bins must match start_prob's duration dimension")
    if any(duration <= 0 for duration in duration_bins):
        raise ValueError("duration_bins must contain positive integers")

    valid_mask = valid_mask_from_lengths(lengths, time, start_prob.device)
    if valid_mask.shape[0] == 1 and batch != 1 and lengths is None:
        valid_mask = valid_mask.expand(batch, -1)
    if valid_mask.shape[0] != batch:
        raise ValueError("lengths batch size does not match start_prob")

    spans = _prepare_spans(span_lengths, valid_mask, start_prob.dtype)
    start_times = spans.cumsum(dim=1) - spans
    valid_count = valid_mask.sum(dim=1).clamp_min(1).to(start_prob.dtype)
    reference_span = spans.sum(dim=1) / valid_count

    # Accumulating log(1 - p) gives a numerically stable noisy-or projection.
    log_not_covered = start_prob.new_zeros(batch, time, start_prob.shape[2])
    target_times = start_times.unsqueeze(1)
    event_starts = start_times.unsqueeze(2)
    pair_is_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
    eps = torch.finfo(start_prob.dtype).eps

    for duration_index, duration in enumerate(duration_bins):
        event_ends = event_starts + reference_span[:, None, None] * duration
        covers_target = (
            (target_times >= event_starts)
            & (target_times < event_ends)
            & pair_is_valid
        )
        log_not_event = torch.log1p(
            -start_prob[:, :, :, duration_index].clamp(max=1.0 - eps)
        )
        log_not_covered = log_not_covered + torch.bmm(
            covers_target.transpose(1, 2).to(start_prob.dtype), log_not_event
        )

    coverage = -torch.expm1(log_not_covered)
    coverage = coverage * valid_mask.unsqueeze(-1).to(coverage.dtype)
    event_score = -torch.expm1(
        torch.log1p(-coverage.clamp(max=1.0 - eps)).sum(dim=-1)
    )
    event_score = event_score * valid_mask.to(event_score.dtype)

    return {
        "coverage": coverage,
        "event_score": event_score,
        "valid_mask": valid_mask,
        "span_lengths": spans,
    }


class CausalEventEncoder(nn.Module):
    """A prefix-only visual encoder for the point-process intensity."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        initial_state: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature]")

        batch, time, _ = features.shape
        encoded = self.input_projection(features)
        if initial_state is not None:
            initial_state = initial_state.to(device=features.device, dtype=features.dtype)
            expected_shape = (1, batch, self.gru.hidden_size)
            if tuple(initial_state.shape) != expected_shape:
                raise ValueError(
                    "initial_state must have shape "
                    f"{expected_shape}, got {tuple(initial_state.shape)}"
                )
        if lengths is None:
            encoded, final_state = self.gru(encoded, initial_state)
            return {"features": encoded, "final_state": final_state}

        valid_mask = valid_mask_from_lengths(lengths, time, features.device)
        if valid_mask.shape[0] != batch:
            raise ValueError("lengths batch size does not match features")

        safe_lengths = valid_mask.sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(
            encoded, safe_lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, final_state = self.gru(packed, initial_state)
        encoded, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=time
        )
        encoded = encoded * valid_mask.unsqueeze(-1).to(encoded.dtype)
        if initial_state is not None:
            final_state = torch.where(
                valid_mask.any(dim=1).view(1, batch, 1), final_state, initial_state
            )
        return {"features": encoded, "final_state": final_state}


class LatentMarkedEventProcess(nn.Module):
    """A discrete hazard approximation to a latent neural marked TPP.

    Each valid input interval can start at most one event.  The event has an anomaly
    class mark and a duration-bin mark.  The intensity at time ``t`` uses a causal
    visual prefix and a recurrent summary of expected events before ``t``.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        duration_bins: Sequence[int],
        hidden_dim: Optional[int] = None,
        mark_feature_dim: Optional[int] = None,
        history_dim: Optional[int] = None,
        use_history: bool = True,
        use_hurdle_gate: bool = False,
        global_hurdle_only: bool = False,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not duration_bins or any(duration <= 0 for duration in duration_bins):
            raise ValueError("duration_bins must be a non-empty positive sequence")

        hidden_dim = hidden_dim or input_dim
        history_dim = history_dim or hidden_dim
        mark_feature_dim = mark_feature_dim or input_dim

        self.num_classes = num_classes
        self.duration_bins = tuple(int(duration) for duration in duration_bins)
        self.use_history = use_history
        self.use_hurdle_gate = use_hurdle_gate
        self.global_hurdle_only = global_hurdle_only
        self.causal_encoder = CausalEventEncoder(input_dim, hidden_dim)
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim + history_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.learned_mark_embeddings = nn.Embedding(num_classes, hidden_dim)
        self.text_mark_projection = nn.Linear(mark_feature_dim, hidden_dim)
        self.duration_embeddings = nn.Embedding(len(self.duration_bins), hidden_dim)
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        # A sparse prior prevents the untrained process from firing in every interval.
        nn.init.constant_(self.intensity_head[-1].bias, -8.5)
        self.event_state_projection = nn.Sequential(
            nn.Linear(hidden_dim, history_dim), nn.Tanh()
        )
        self.history_cell = nn.GRUCell(history_dim, history_dim)
        if self.use_hurdle_gate:
            self.hurdle_attention = nn.Linear(hidden_dim, 1)
            self.hurdle_context = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
            )
            if not self.global_hurdle_only:
                self.hurdle_bias = nn.Parameter(torch.full((num_classes,), -2.0))
            self.hurdle_global_head = nn.Linear(hidden_dim, 1)
            nn.init.constant_(self.hurdle_global_head.bias, -2.0)

    def _mark_vectors(self, mark_features: Optional[Tensor]) -> Tensor:
        learned = self.learned_mark_embeddings.weight
        if mark_features is None:
            return learned
        if mark_features.ndim != 2 or mark_features.shape[0] != self.num_classes:
            raise ValueError(
                "mark_features must have shape [num_classes, mark_feature_dim]"
            )
        if mark_features.shape[1] != self.text_mark_projection.in_features:
            raise ValueError("mark_features has an unexpected embedding dimension")
        projected = self.text_mark_projection(
            mark_features.to(device=learned.device, dtype=learned.dtype)
        )
        return learned + projected

    def _hurdle_probabilities(
        self, causal_features: Tensor, valid_mask: Tensor, mark_vectors: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
        """Estimate global and class-specific activation of the event process."""

        attention_mask = valid_mask.clone()
        attention_mask[~attention_mask.any(dim=1), 0] = True
        attention_logits = self.hurdle_attention(causal_features).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~attention_mask, -torch.inf)
        attention = torch.softmax(attention_logits, dim=1)
        pooled = torch.einsum("bt,bth->bh", attention, causal_features)
        context = self.hurdle_context(pooled)
        if self.global_hurdle_only:
            class_logits = None
            class_presence_prob = context.new_ones(context.shape[0], self.num_classes)
        else:
            class_logits = context @ mark_vectors.transpose(0, 1)
            class_logits = class_logits / (context.shape[-1] ** 0.5) + self.hurdle_bias
            class_presence_prob = torch.sigmoid(class_logits)
        global_logits = self.hurdle_global_head(context).squeeze(-1)
        return (
            global_logits,
            torch.sigmoid(global_logits),
            class_logits,
            class_presence_prob,
        )

    def encode_causal_prefix(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        initial_causal_state: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Encode a prefix without sampling events or applying a Hurdle gate.

        Offline inference uses this pass to aggregate one video-level gate across
        multiple model-sized chunks.  The returned causal features are independent of
        event history, so concatenating prefix outputs is equivalent to one unbroken
        causal encoding.
        """

        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature]")

        features = features.float()
        batch, time, _ = features.shape
        valid_mask = valid_mask_from_lengths(lengths, time, features.device)
        if valid_mask.shape[0] == 1 and batch != 1 and lengths is None:
            valid_mask = valid_mask.expand(batch, -1)
        if valid_mask.shape[0] != batch:
            raise ValueError("lengths batch size does not match features")

        causal_output = self.causal_encoder(
            features, lengths, initial_state=initial_causal_state
        )
        return {
            "features": causal_output["features"],
            "valid_mask": valid_mask,
            "final_state": causal_output["final_state"],
        }

    def hurdle_from_causal_features(
        self,
        causal_features: Tensor,
        valid_mask: Tensor,
        mark_features: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute one Hurdle gate from an already concatenated video prefix."""

        if not self.use_hurdle_gate:
            batch = causal_features.shape[0]
            ones_global = causal_features.new_ones(batch)
            ones_class = causal_features.new_ones(batch, self.num_classes)
            zeros_global = causal_features.new_zeros(batch)
            zeros_class = causal_features.new_zeros(batch, self.num_classes)
            return zeros_global, ones_global, zeros_class, ones_class

        mark_vectors = self._mark_vectors(mark_features)
        return self._hurdle_probabilities(causal_features, valid_mask, mark_vectors)

    def forward(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        span_lengths: Optional[Tensor] = None,
        mark_features: Optional[Tensor] = None,
        initial_history: Optional[Tensor] = None,
        initial_causal_state: Optional[Tensor] = None,
        hurdle_override: Optional[tuple[Tensor, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature]")

        features = features.float()
        batch, time, _ = features.shape
        valid_mask = valid_mask_from_lengths(lengths, time, features.device)
        if valid_mask.shape[0] == 1 and batch != 1 and lengths is None:
            valid_mask = valid_mask.expand(batch, -1)
        if valid_mask.shape[0] != batch:
            raise ValueError("lengths batch size does not match features")

        spans = _prepare_spans(span_lengths, valid_mask, features.dtype)
        causal_output = self.causal_encoder(
            features, lengths, initial_state=initial_causal_state
        )
        causal_features = causal_output["features"]
        mark_vectors = self._mark_vectors(mark_features)
        if self.use_hurdle_gate:
            if hurdle_override is None:
                (
                    hurdle_global_logits,
                    hurdle_global_presence_prob,
                    hurdle_logits,
                    hurdle_presence_prob,
                ) = self._hurdle_probabilities(causal_features, valid_mask, mark_vectors)
            else:
                if len(hurdle_override) != 2:
                    raise ValueError(
                        "hurdle_override must contain global and class presence probabilities"
                    )
                hurdle_global_presence_prob, hurdle_presence_prob = (
                    tensor.to(device=features.device, dtype=features.dtype)
                    for tensor in hurdle_override
                )
                expected_global = (batch,)
                expected_class = (batch, self.num_classes)
                if tuple(hurdle_global_presence_prob.shape) != expected_global:
                    raise ValueError(
                        f"hurdle global override must have shape {expected_global}"
                    )
                if tuple(hurdle_presence_prob.shape) != expected_class:
                    raise ValueError(
                        f"hurdle class override must have shape {expected_class}"
                    )
                eps_gate = torch.finfo(features.dtype).eps
                hurdle_global_presence_prob = hurdle_global_presence_prob.clamp(
                    min=eps_gate, max=1.0 - eps_gate
                )
                hurdle_global_logits = torch.logit(hurdle_global_presence_prob)
                if self.global_hurdle_only:
                    hurdle_logits = None
                    hurdle_presence_prob = features.new_ones(batch, self.num_classes)
                else:
                    hurdle_presence_prob = hurdle_presence_prob.clamp(
                        min=eps_gate, max=1.0 - eps_gate
                    )
                    hurdle_logits = torch.logit(hurdle_presence_prob)
        else:
            hurdle_global_logits = None
            hurdle_global_presence_prob = features.new_ones(batch)
            hurdle_logits = None
            hurdle_presence_prob = features.new_ones(batch, self.num_classes)
        duration_vectors = self.duration_embeddings.weight
        mark_duration_vectors = mark_vectors[:, None, :] + duration_vectors[None, :, :]
        event_state_vectors = self.event_state_projection(mark_duration_vectors)

        if initial_history is None:
            history = features.new_zeros(batch, self.history_cell.hidden_size)
        else:
            history = initial_history.to(device=features.device, dtype=features.dtype)
            expected_shape = (batch, self.history_cell.hidden_size)
            if tuple(history.shape) != expected_shape:
                raise ValueError(
                    "initial_history must have shape "
                    f"{expected_shape}, got {tuple(history.shape)}"
                )
        if not self.use_history:
            history = torch.zeros_like(history)
        intensities = []
        start_probabilities = []
        conditional_start_probabilities = []
        empty_probabilities = []
        eps = torch.finfo(features.dtype).eps

        for step in range(time):
            valid_step = valid_mask[:, step]
            context = self.context_projection(
                torch.cat([causal_features[:, step], history], dim=-1)
            )
            event_context = (
                context[:, None, None, :]
                + mark_vectors[None, :, None, :]
                + duration_vectors[None, None, :, :]
            )
            intensity = F.softplus(self.intensity_head(event_context).squeeze(-1))
            intensity = intensity * valid_step[:, None, None].to(intensity.dtype)
            total_intensity = intensity.sum(dim=(1, 2))
            event_mass = -torch.expm1(-total_intensity * spans[:, step])
            conditional_start_probability = (
                event_mass[:, None, None]
                * intensity
                / total_intensity[:, None, None].clamp_min(eps)
            )
            conditional_start_probability = conditional_start_probability * valid_step[:, None, None].to(
                conditional_start_probability.dtype
            )
            start_probability = conditional_start_probability * (
                hurdle_global_presence_prob[:, None, None]
                * hurdle_presence_prob[:, :, None]
            )
            if self.use_hurdle_gate:
                empty_probability = 1.0 - start_probability.sum(dim=(1, 2))
            else:
                empty_probability = torch.exp(-total_intensity * spans[:, step])
            empty_probability = torch.where(
                valid_step, empty_probability, torch.ones_like(empty_probability)
            )

            expected_event = torch.einsum(
                "bcd,cdh->bh", start_probability, event_state_vectors
            )
            if self.use_history:
                updated_history = self.history_cell(expected_event, history)
                history = torch.where(valid_step[:, None], updated_history, history)

            intensities.append(intensity)
            start_probabilities.append(start_probability)
            conditional_start_probabilities.append(conditional_start_probability)
            empty_probabilities.append(empty_probability)

        intensity = torch.stack(intensities, dim=1)
        start_prob = torch.stack(start_probabilities, dim=1)
        conditional_start_prob = torch.stack(conditional_start_probabilities, dim=1)
        empty_prob = torch.stack(empty_probabilities, dim=1)
        conditional_expected_count = (
            intensity * spans[:, :, None, None]
        ).sum(dim=(1, 3))
        expected_count = conditional_expected_count * (
            hurdle_global_presence_prob[:, None] * hurdle_presence_prob
        )
        coverage_output = project_event_coverage(
            start_prob,
            self.duration_bins,
            lengths=lengths,
            span_lengths=spans,
        )

        output = {
            "intensity": intensity,
            "start_prob": start_prob,
            "conditional_start_prob": conditional_start_prob,
            "empty_prob": empty_prob,
            "expected_count": expected_count,
            "conditional_expected_count": conditional_expected_count,
            "final_history": history,
            "final_causal_state": causal_output["final_state"],
            **coverage_output,
        }
        if self.use_hurdle_gate:
            output["hurdle_global_logits"] = hurdle_global_logits
            output["hurdle_global_presence_prob"] = hurdle_global_presence_prob
            output["hurdle_logits"] = hurdle_logits
            output["hurdle_presence_prob"] = hurdle_presence_prob
        return output


def tracepoint_weak_loss(
    output: Dict[str, Tensor],
    labels: Tensor,
    count_cap: float = 2.0,
    duration_entropy_fraction: float = 0.3,
    count_weight: float = 0.05,
    duration_weight: float = 0.01,
    fragment_weight: float = 0.01,
    align_weight: float = 0.0,
    hurdle_weight: float = 1.0,
    normal_zero_weight: float = 0.0,
    normal_zero_gate_only: bool = False,
) -> Dict[str, Tensor]:
    """Compute weak bag-label supervision for the latent event process.

    ``labels`` can include a leading normal column, as produced by VadCLIP's label
    helper, or contain only anomaly classes.
    """

    expected_count = output["expected_count"]
    labels = labels.to(expected_count.device, dtype=expected_count.dtype)
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, class]")
    if labels.shape[1] == expected_count.shape[1] + 1:
        targets = labels[:, 1:]
    elif labels.shape[1] == expected_count.shape[1]:
        targets = labels
    else:
        raise ValueError("labels does not match the process class dimension")
    targets = targets.clamp(min=0.0, max=1.0)

    eps = torch.finfo(expected_count.dtype).eps
    positive_denominator = targets.sum().clamp_min(1.0)
    absent_targets = 1.0 - targets
    absent_denominator = absent_targets.sum().clamp_min(1.0)
    hurdle_loss = expected_count.new_zeros(())
    if "hurdle_presence_prob" in output:
        conditional_presence = -torch.expm1(-output["conditional_expected_count"])
        event_presence = (
            output["hurdle_global_presence_prob"][:, None]
            * output["hurdle_presence_prob"]
            * conditional_presence
        ).clamp(min=eps, max=1.0 - eps)
        presence_loss = -(targets * event_presence.log()).sum() / positive_denominator
        absence_loss = -(
            absent_targets * torch.log1p(-event_presence)
        ).sum() / absent_denominator

        global_targets = targets.any(dim=1).to(event_presence.dtype)
        global_presence = output["hurdle_global_presence_prob"].clamp(
            min=eps, max=1.0 - eps
        )
        global_positive_denominator = global_targets.sum().clamp_min(1.0)
        global_absent_denominator = (1.0 - global_targets).sum().clamp_min(1.0)
        global_hurdle_loss = -(
            global_targets * global_presence.log()
        ).sum() / global_positive_denominator
        global_hurdle_loss = global_hurdle_loss - (
            (1.0 - global_targets) * torch.log1p(-global_presence)
        ).sum() / global_absent_denominator
        if output.get("hurdle_logits") is None:
            hurdle_loss = global_hurdle_loss
        else:
            hurdle_presence = output["hurdle_presence_prob"].clamp(
                min=eps, max=1.0 - eps
            )
            hurdle_positive = -(targets * hurdle_presence.log()).sum() / positive_denominator
            hurdle_absence = -(
                absent_targets * torch.log1p(-hurdle_presence)
            ).sum() / absent_denominator
            hurdle_loss = hurdle_positive + hurdle_absence + global_hurdle_loss
    else:
        log_presence = torch.log((-torch.expm1(-expected_count)).clamp_min(eps))
        presence_loss = -(targets * log_presence).sum() / positive_denominator
        absence_loss = (absent_targets * expected_count).sum() / absent_denominator
    # ``event_loss`` is the weak event-presence likelihood after marginalizing the
    # conditional process.  For a Hurdle model, ``hurdle_loss`` below is deliberately
    # kept as an explicit auxiliary BCE on its gate(s); it is not another term in the
    # claimed probability factorization.
    event_loss = presence_loss + absence_loss

    positive_videos = targets.any(dim=1)
    total_count = expected_count.sum(dim=1)
    count_loss = (
        F.relu(total_count - count_cap).square() * positive_videos.to(total_count.dtype)
    ).sum() / positive_videos.sum().clamp_min(1).to(total_count.dtype)

    duration_mass = output["start_prob"].sum(dim=(0, 1, 2))
    duration_total = duration_mass.sum()
    duration_distribution = duration_mass / duration_total.clamp_min(eps)
    duration_entropy = -(
        duration_distribution * duration_distribution.clamp_min(eps).log()
    ).sum()
    target_entropy = duration_entropy_fraction * torch.log(
        torch.tensor(
            float(duration_mass.numel()), device=duration_entropy.device, dtype=duration_entropy.dtype
        )
    )
    duration_loss = F.relu(target_entropy - duration_entropy)
    duration_loss = duration_loss * (duration_total.detach() > eps).to(duration_loss.dtype)

    event_score = output["event_score"]
    valid_mask = output["valid_mask"]
    valid_pairs = valid_mask[:, 1:] & valid_mask[:, :-1]
    fragment_loss = (
        (event_score[:, 1:] - event_score[:, :-1]).abs()
        * valid_pairs.to(event_score.dtype)
    ).sum() / valid_pairs.sum().clamp_min(1).to(event_score.dtype)

    normal_zero_loss = event_score.new_zeros(())
    if normal_zero_weight:
        normal_videos = ~positive_videos
        if normal_zero_gate_only:
            if "hurdle_presence_prob" not in output:
                raise ValueError(
                    "normal_zero_gate_only requires a Hurdle-TracePoint output"
                )
            conditional_presence = -torch.expm1(
                -output["conditional_expected_count"].detach()
            )
            gate_presence = (
                output["hurdle_global_presence_prob"][:, None]
                * output["hurdle_presence_prob"]
                * conditional_presence
            ).clamp(max=1.0 - eps)
            normal_weights = normal_videos[:, None].to(event_score.dtype)
            normal_zero_loss = -(
                torch.log1p(-gate_presence) * normal_weights
            ).sum() / normal_weights.sum().clamp_min(1)
        else:
            normal_weights = (
                normal_videos[:, None] & valid_mask
            ).to(event_score.dtype)
            normal_zero_loss = -(
                torch.log1p(-event_score.clamp(max=1.0 - eps)) * normal_weights
            ).sum() / normal_weights.sum().clamp_min(1)

    align_loss = event_score.new_zeros(())
    if align_weight and "baseline_score" in output:
        per_step = F.smooth_l1_loss(
            event_score, output["baseline_score"].detach(), reduction="none"
        )
        align_loss = (per_step * valid_mask.to(per_step.dtype)).sum() / valid_mask.sum().clamp_min(1).to(per_step.dtype)

    total = (
        event_loss
        + count_weight * count_loss
        + duration_weight * duration_loss
        + fragment_weight * fragment_loss
        + align_weight * align_loss
        + hurdle_weight * hurdle_loss
        + normal_zero_weight * normal_zero_loss
    )
    return {
        "total": total,
        "event": event_loss,
        "presence": presence_loss,
        "absence": absence_loss,
        "count": count_loss,
        "duration": duration_loss,
        "fragment": fragment_loss,
        "align": align_loss,
        "hurdle": hurdle_loss,
        "normal_zero": normal_zero_loss,
    }


def infer_tracepoint_chunks(
    model,
    visual: Tensor,
    padding_mask: Optional[Tensor],
    lengths: Tensor,
    spans: Tensor,
    prompt_text,
) -> Dict[str, Optional[Tensor]]:
    """Run one video through model-sized chunks with a video-level Hurdle gate.

    The Hurdle gate is defined over the complete video, while the causal event process
    can be evaluated sequentially.  A prefix pass therefore computes the gate once and
    a second pass carries the event/history states through the chunks.
    """

    if not getattr(model, "tracepoint_enabled", False) or model.tracepoint is None:
        raise ValueError("infer_tracepoint_chunks requires an enabled TracePoint branch")

    tracepoint = model.tracepoint
    chunk_count = visual.shape[0]
    if spans.shape[0] != chunk_count or lengths.shape[0] != chunk_count:
        raise ValueError("visual, spans, and lengths must have the same chunk count")

    hurdle_presence_prob = None
    hurdle_global_presence_prob = None
    hurdle_override = None

    if tracepoint.use_hurdle_gate:
        # The video-level gate reads semantic anchors but never owns or updates the
        # primary prompt path, including when this helper is reused outside eval.
        mark_features = model.encode_textprompt(prompt_text)[1:].detach()
        causal_state = None
        prefix_features = []
        prefix_masks = []
        for chunk in range(chunk_count):
            prefix = tracepoint.encode_causal_prefix(
                visual[chunk : chunk + 1],
                lengths[chunk : chunk + 1],
                initial_causal_state=causal_state,
            )
            causal_state = prefix["final_state"]
            prefix_features.append(prefix["features"])
            prefix_masks.append(prefix["valid_mask"])

        causal_features = torch.cat(prefix_features, dim=1)
        valid_mask = torch.cat(prefix_masks, dim=1)
        (
            _,
            hurdle_global_presence_prob,
            _,
            hurdle_presence_prob,
        ) = tracepoint.hurdle_from_causal_features(
            causal_features, valid_mask, mark_features
        )
        hurdle_override = (hurdle_global_presence_prob, hurdle_presence_prob)

    history = None
    causal_state = None
    logits1_chunks = []
    logits2_chunks = []
    start_prob_chunks = []
    span_chunks = []
    for chunk in range(chunk_count):
        _, current_logits1, current_logits2, current_output = model(
            visual[chunk : chunk + 1],
            padding_mask[chunk : chunk + 1] if padding_mask is not None else None,
            prompt_text,
            lengths[chunk : chunk + 1],
            span_lengths=spans[chunk : chunk + 1],
            return_tracepoint=True,
            tracepoint_history=history,
            tracepoint_causal_state=causal_state,
            hurdle_override=hurdle_override,
        )
        history = current_output["final_history"]
        causal_state = current_output["final_causal_state"]
        logits1_chunks.append(current_logits1)
        logits2_chunks.append(current_logits2)
        start_prob_chunks.append(current_output["start_prob"])
        span_chunks.append(current_output["span_lengths"])

    return {
        "logits1": torch.cat(logits1_chunks, dim=0),
        "logits2": torch.cat(logits2_chunks, dim=0),
        "start_prob": torch.cat(start_prob_chunks, dim=1),
        "span_lengths": torch.cat(span_chunks, dim=1),
        "hurdle_presence_prob": hurdle_presence_prob,
        "hurdle_global_presence_prob": hurdle_global_presence_prob,
    }
