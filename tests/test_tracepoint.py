import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from tracepoint import (
    LatentMarkedEventProcess,
    TracePointDiagnostics,
    project_event_coverage,
    tracepoint_weak_loss,
)
from utils.checkpoint import load_checkpoint_payload
from utils.experiment import (
    append_metrics,
    configure_run_paths,
    freeze_to_tracepoint_branch,
    resolve_selection_metric,
)
from utils import tools


class TracePointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.process = LatentMarkedEventProcess(
            input_dim=8,
            num_classes=3,
            duration_bins=(1, 2, 4),
            hidden_dim=12,
            mark_feature_dim=8,
        )
        self.features = torch.randn(2, 5, 8)
        self.lengths = torch.tensor([5, 3])
        self.spans = torch.tensor([[1.0, 1.0, 2.0, 1.0, 1.0], [1.0, 2.0, 1.0, 0.0, 0.0]])
        self.mark_features = torch.randn(3, 8)

    def test_hazard_probabilities_and_padding(self):
        output = self.process(
            self.features, self.lengths, self.spans, self.mark_features
        )
        total_probability = output["start_prob"].sum(dim=(2, 3)) + output["empty_prob"]
        torch.testing.assert_close(total_probability, torch.ones_like(total_probability), atol=1e-6, rtol=0)
        self.assertTrue(torch.equal(output["intensity"][1, 3:], torch.zeros_like(output["intensity"][1, 3:])))
        self.assertTrue(torch.equal(output["coverage"][1, 3:], torch.zeros_like(output["coverage"][1, 3:])))

    def test_future_features_do_not_change_past_intensity(self):
        self.process.eval()
        changed_features = self.features.clone()
        changed_features[:, 4] = changed_features[:, 4] + 100.0
        first = self.process(self.features, self.lengths, self.spans, self.mark_features)
        second = self.process(changed_features, self.lengths, self.spans, self.mark_features)
        torch.testing.assert_close(first["intensity"][:, :4], second["intensity"][:, :4], atol=1e-6, rtol=0)

    def test_chunk_states_match_an_unbroken_event_prefix(self):
        self.process.eval()
        features = self.features[:1]
        spans = self.spans[:1]
        full = self.process(features, torch.tensor([5]), spans, self.mark_features)
        first_chunk = self.process(
            features[:, :3], torch.tensor([3]), spans[:, :3], self.mark_features
        )
        second_chunk = self.process(
            features[:, 3:],
            torch.tensor([2]),
            spans[:, 3:],
            self.mark_features,
            initial_history=first_chunk["final_history"],
            initial_causal_state=first_chunk["final_causal_state"],
        )
        torch.testing.assert_close(
            full["intensity"][:, 3:], second_chunk["intensity"], atol=1e-6, rtol=0
        )

    def test_video_hurdle_gate_matches_concatenated_chunk_features(self):
        process = LatentMarkedEventProcess(
            input_dim=8,
            num_classes=3,
            duration_bins=(1, 2, 4),
            hidden_dim=12,
            mark_feature_dim=8,
            use_hurdle_gate=True,
        ).eval()
        features = self.features[:1]
        spans = self.spans[:1]
        marks = self.mark_features
        full = process(features, torch.tensor([5]), spans, marks)

        first_prefix = process.encode_causal_prefix(
            features[:, :3], torch.tensor([3]), initial_causal_state=None
        )
        second_prefix = process.encode_causal_prefix(
            features[:, 3:],
            torch.tensor([2]),
            initial_causal_state=first_prefix["final_state"],
        )
        causal_features = torch.cat(
            [first_prefix["features"], second_prefix["features"]], dim=1
        )
        valid_mask = torch.cat(
            [first_prefix["valid_mask"], second_prefix["valid_mask"]], dim=1
        )
        _, global_presence, _, class_presence = process.hurdle_from_causal_features(
            causal_features, valid_mask, marks
        )
        torch.testing.assert_close(
            global_presence,
            full["hurdle_global_presence_prob"],
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            class_presence,
            full["hurdle_presence_prob"],
            atol=1e-6,
            rtol=0,
        )

        first_events = process(
            features[:, :3],
            torch.tensor([3]),
            spans[:, :3],
            marks,
            hurdle_override=(global_presence, class_presence),
        )
        second_events = process(
            features[:, 3:],
            torch.tensor([2]),
            spans[:, 3:],
            marks,
            initial_history=first_events["final_history"],
            initial_causal_state=first_events["final_causal_state"],
            hurdle_override=(global_presence, class_presence),
        )
        torch.testing.assert_close(
            first_events["start_prob"], full["start_prob"][:, :3], atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            second_events["start_prob"], full["start_prob"][:, 3:], atol=1e-6, rtol=0
        )

    def test_history_free_ablation_ignores_carried_event_state(self):
        process = LatentMarkedEventProcess(
            input_dim=8,
            num_classes=3,
            duration_bins=(1, 2, 4),
            hidden_dim=12,
            mark_feature_dim=8,
            use_history=False,
        ).eval()
        features = self.features[:1]
        lengths = torch.tensor([5])
        spans = self.spans[:1]
        no_state = process(features, lengths, spans, self.mark_features)
        carried_state = process(
            features,
            lengths,
            spans,
            self.mark_features,
            initial_history=torch.randn(1, 12),
        )
        torch.testing.assert_close(
            no_state["intensity"], carried_state["intensity"], atol=1e-6, rtol=0
        )
        self.assertTrue(
            torch.equal(no_state["final_history"], torch.zeros_like(no_state["final_history"]))
        )

    def test_duration_projection_covers_the_expected_interval(self):
        starts = torch.zeros(1, 4, 1, 3)
        starts[0, 0, 0, 1] = 0.7
        output = project_event_coverage(
            starts,
            duration_bins=(1, 2, 4),
            lengths=torch.tensor([4]),
            span_lengths=torch.ones(1, 4),
        )
        expected = torch.tensor([0.7, 0.7, 0.0, 0.0])
        torch.testing.assert_close(output["coverage"][0, :, 0], expected, atol=1e-6, rtol=0)

    def test_global_duration_projection_crosses_a_chunk_boundary(self):
        first_chunk_starts = torch.zeros(1, 4, 1, 1)
        second_chunk_starts = torch.zeros(1, 2, 1, 1)
        first_chunk_starts[0, 3, 0, 0] = 0.6

        output = project_event_coverage(
            torch.cat([first_chunk_starts, second_chunk_starts], dim=1),
            duration_bins=(2,),
            lengths=torch.tensor([6]),
            span_lengths=torch.ones(1, 6),
        )
        expected = torch.tensor([0.0, 0.0, 0.0, 0.6, 0.6, 0.0])
        torch.testing.assert_close(output["coverage"][0, :, 0], expected, atol=1e-6, rtol=0)

    def test_weak_loss_backpropagates(self):
        output = self.process(
            self.features, self.lengths, self.spans, self.mark_features
        )
        labels = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
        losses = tracepoint_weak_loss(output, labels)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(self.process.intensity_head[-1].bias.grad)
        self.assertTrue(torch.isfinite(self.process.intensity_head[-1].bias.grad).all())

    def test_hurdle_gate_forms_a_valid_event_distribution_and_backpropagates(self):
        process = LatentMarkedEventProcess(
            input_dim=8,
            num_classes=3,
            duration_bins=(1, 2, 4),
            hidden_dim=12,
            mark_feature_dim=8,
            use_hurdle_gate=True,
        )
        output = process(self.features, self.lengths, self.spans, self.mark_features)
        total_probability = output['start_prob'].sum(dim=(2, 3)) + output['empty_prob']
        torch.testing.assert_close(
            total_probability, torch.ones_like(total_probability), atol=1e-6, rtol=0
        )
        self.assertTrue(torch.all(output['hurdle_presence_prob'] > 0))
        self.assertTrue(torch.all(output['hurdle_presence_prob'] < 1))
        self.assertTrue(torch.all(output['hurdle_global_presence_prob'] > 0))
        self.assertTrue(torch.all(output['hurdle_global_presence_prob'] < 1))

        labels = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
        losses = tracepoint_weak_loss(output, labels, hurdle_weight=1.0)
        losses['total'].backward()
        self.assertIsNotNone(process.hurdle_bias.grad)
        self.assertTrue(torch.isfinite(process.hurdle_bias.grad).all())

    def test_global_only_hurdle_uses_a_binary_structural_gate(self):
        process = LatentMarkedEventProcess(
            input_dim=8,
            num_classes=3,
            duration_bins=(1, 2, 4),
            hidden_dim=12,
            mark_feature_dim=8,
            use_hurdle_gate=True,
            global_hurdle_only=True,
        )
        output = process(self.features, self.lengths, self.spans, self.mark_features)
        torch.testing.assert_close(
            output['hurdle_presence_prob'],
            torch.ones_like(output['hurdle_presence_prob']),
            atol=0,
            rtol=0,
        )
        self.assertIsNone(output['hurdle_logits'])

        labels = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
        losses = tracepoint_weak_loss(output, labels, hurdle_weight=1.0)
        self.assertTrue(torch.isfinite(losses['total']))
        losses['total'].backward()
        self.assertIsNotNone(process.hurdle_global_head.weight.grad)
        self.assertTrue(torch.isfinite(process.hurdle_global_head.weight.grad).all())

    def test_structural_zero_coverage_penalty_reaches_hurdle_process(self):
        process = LatentMarkedEventProcess(
            input_dim=8,
            num_classes=3,
            duration_bins=(1, 2, 4),
            hidden_dim=12,
            mark_feature_dim=8,
            use_hurdle_gate=True,
        )
        output = process(self.features, self.lengths, self.spans, self.mark_features)
        labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        losses = tracepoint_weak_loss(output, labels, normal_zero_weight=0.5)
        self.assertGreater(float(losses['normal_zero'].detach()), 0.0)
        losses['total'].backward()
        self.assertIsNotNone(process.hurdle_global_head.weight.grad)
        self.assertTrue(torch.isfinite(process.hurdle_global_head.weight.grad).all())

    def test_gate_only_zero_loss_does_not_directly_update_intensity_head(self):
        process = LatentMarkedEventProcess(
            input_dim=8,
            num_classes=3,
            duration_bins=(1, 2, 4),
            hidden_dim=12,
            mark_feature_dim=8,
            use_hurdle_gate=True,
        )
        output = process(self.features, self.lengths, self.spans, self.mark_features)
        labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        losses = tracepoint_weak_loss(
            output,
            labels,
            normal_zero_weight=0.5,
            normal_zero_gate_only=True,
        )
        self.assertGreater(float(losses['normal_zero'].detach()), 0.0)
        losses['normal_zero'].backward()
        self.assertIsNotNone(process.hurdle_global_head.weight.grad)
        self.assertIsNone(process.intensity_head[-1].bias.grad)

    def test_span_helpers_preserve_source_extent(self):
        features = np.arange(10, dtype=np.float32).reshape(10, 1)
        _, valid_length, spans = tools.process_feat_with_spans(features, 4)
        self.assertEqual(valid_length, 4)
        np.testing.assert_array_equal(spans, np.array([2.0, 3.0, 2.0, 3.0], dtype=np.float32))
        self.assertEqual(spans.sum(), 10.0)

        chunks, original_length, split_spans = tools.process_split_with_spans(features[:6], 4)
        self.assertEqual(original_length, 6)
        self.assertEqual(chunks.shape, (2, 4, 1))
        np.testing.assert_array_equal(split_spans[1], np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))

        exact_chunks, exact_length, exact_spans = tools.process_split_with_spans(features[:8], 4)
        self.assertEqual(exact_length, 8)
        self.assertEqual(exact_chunks.shape, (2, 4, 1))
        np.testing.assert_array_equal(exact_spans, np.ones((2, 4), dtype=np.float32))

        baseline_chunks, baseline_length = tools.process_split(features[:8], 4)
        self.assertEqual(baseline_length, 8)
        self.assertEqual(baseline_chunks.shape, (2, 4, 1))

    def test_checkpoint_loader_accepts_raw_state_and_training_checkpoint(self):
        state = {'weight': torch.randn(2, 3)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / 'raw.pth'
            checkpoint_path = root / 'checkpoint.pth'
            torch.save(state, raw_path)
            torch.save({'model_state_dict': state, 'epoch': 4}, checkpoint_path)

            raw = load_checkpoint_payload(raw_path, map_location='cpu')
            checkpoint = load_checkpoint_payload(checkpoint_path, map_location='cpu')

        self.assertEqual(raw.state_format, 'full_model_state')
        self.assertEqual(checkpoint.state_format, 'training_checkpoint')
        torch.testing.assert_close(raw.state_dict['weight'], state['weight'])
        torch.testing.assert_close(checkpoint.state_dict['weight'], state['weight'])

    def test_diagnostics_distinguishes_normal_and_positive_event_mass(self):
        diagnostics = TracePointDiagnostics((1, 2))
        start_prob = torch.zeros(1, 2, 1, 2)
        start_prob[0, 0, 0, 0] = 0.2
        start_prob[0, 1, 0, 1] = 0.4
        diagnostics.update(torch.tensor([[0.2, 0.4]]), start_prob, is_normal=False)
        diagnostics.update(torch.tensor([[0.1, 0.1]]), start_prob * 0.5, is_normal=True)

        summary = diagnostics.as_dict()
        self.assertAlmostEqual(summary['positive_mean_event_score'], 0.3)
        self.assertAlmostEqual(summary['normal_mean_event_score'], 0.1)
        self.assertAlmostEqual(summary['positive_mean_expected_starts'], 0.6)
        self.assertAlmostEqual(summary['normal_mean_expected_starts'], 0.3)
        self.assertEqual(summary['duration_bins'], [1, 2])
        self.assertAlmostEqual(sum(summary['duration_distribution']), 1.0)

    def test_output_directory_uses_one_best_checkpoint_and_jsonl_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                output_dir=directory,
                checkpoint_path='unused/checkpoint.pth',
                model_path='unused/model.pth',
                current_model_path='unused/current.pth',
                metrics_path=None,
                seed=7,
            )
            configure_run_paths(args, 'ucf')
            self.assertEqual(args.checkpoint_path, args.model_path)
            append_metrics(args.metrics_path, {'auc': 0.5})
            metrics = Path(args.metrics_path).read_text(encoding='utf-8').strip()
            self.assertEqual(metrics, '{"auc": 0.5}')

    def test_selection_metric_supports_high_iou_localization(self):
        metrics = {'auc': 0.8, 'localization_map': {'0.5': 12.5}}
        self.assertEqual(resolve_selection_metric(metrics, 'auc'), 0.8)
        self.assertEqual(resolve_selection_metric(metrics, 'map_0.5'), 12.5)

    def test_tracepoint_only_freeze_leaves_only_the_add_on_trainable(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(2, 2)
                self.tracepoint = torch.nn.Linear(2, 2)
                self.tracepoint_fusion_logit = torch.nn.Parameter(torch.tensor(-5.0))

        model = ToyModel()
        counts = freeze_to_tracepoint_branch(model)

        self.assertFalse(model.backbone.weight.requires_grad)
        self.assertFalse(model.backbone.bias.requires_grad)
        self.assertTrue(model.tracepoint.weight.requires_grad)
        self.assertTrue(model.tracepoint.bias.requires_grad)
        self.assertTrue(model.tracepoint_fusion_logit.requires_grad)
        self.assertEqual(counts['trainable'], 7)
        self.assertEqual(counts['frozen'], 6)

    def test_metrics_only_pilot_mode_is_recorded_in_a_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                output_dir=directory,
                checkpoint_path='unused/checkpoint.pth',
                model_path='unused/model.pth',
                current_model_path='unused/current.pth',
                metrics_path=None,
                seed=7,
                tracepoint_save_checkpoint=False,
            )
            configure_run_paths(args, 'ucf')
            manifest = Path(directory, 'run_config.json').read_text(encoding='utf-8')

        self.assertIn('"tracepoint_save_checkpoint": false', manifest)


if __name__ == "__main__":
    unittest.main()
