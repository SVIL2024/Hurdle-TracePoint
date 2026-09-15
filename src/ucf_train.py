import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
from pathlib import Path

from model import CLIPVAD
from tracepoint import tracepoint_weak_loss
from ucf_test import test
from utils.checkpoint import load_checkpoint_payload
from utils.dataset import UCFDataset
from utils.experiment import (
    append_metrics,
    configure_run_paths,
    isolated_parameter_groups,
    resolve_selection_metric,
    validate_isolated_tracepoint_args,
)
from utils.tools import get_prompt_text, get_batch_label
import ucf_option

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


def load_initial_state(model, primary_optimizer, event_optimizer, args, device):
    if args.warm_start_path and args.use_checkpoint:
        raise ValueError('use either --warm-start-path or --use-checkpoint, not both')

    source = args.warm_start_path
    resume_requested = False
    if source is None and args.use_checkpoint:
        source = args.checkpoint_path
        resume_requested = True
    if source is None:
        return 0, float('-inf')

    payload = load_checkpoint_payload(source, map_location=device)
    incompatible = model.load_state_dict(
        payload.state_dict,
        strict=not args.tracepoint,
    )
    if args.tracepoint and incompatible.missing_keys:
        print('warm-start missing keys:', len(incompatible.missing_keys))
    if args.tracepoint and incompatible.unexpected_keys:
        print('warm-start unexpected keys:', len(incompatible.unexpected_keys))

    can_resume = (
        resume_requested
        and payload.state_format == 'training_checkpoint'
        and payload.has_optimizer_state
        and (not args.tracepoint or payload.has_tracepoint_state)
    )
    if can_resume:
        if event_optimizer is None:
            primary_optimizer.load_state_dict(payload.metadata['optimizer_state_dict'])
        else:
            primary_state = payload.metadata.get('primary_optimizer_state_dict')
            event_state = payload.metadata.get('event_optimizer_state_dict')
            if primary_state is None or event_state is None:
                raise ValueError('isolated TracePoint resume requires both optimizer states')
            primary_optimizer.load_state_dict(primary_state)
            event_optimizer.load_state_dict(event_state)
        start_epoch = int(payload.metadata.get('epoch', -1)) + 1
        best_metric = float(
            payload.metadata.get('primary_metric', payload.metadata.get('ap', float('-inf')))
        )
        print('resumed checkpoint:', source, '| next epoch:', start_epoch + 1, '| best:', best_metric)
        return start_epoch, best_metric

    print('warm-started model:', source, '| format:', payload.state_format, '| optimizer: fresh')
    return 0, float('-inf')


def save_best_checkpoint(
    model,
    primary_optimizer,
    event_optimizer,
    args,
    epoch,
    primary_metric,
    metrics,
):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ap': primary_metric,
        'primary_metric': primary_metric,
        'metrics': metrics,
    }
    if event_optimizer is None:
        checkpoint['optimizer_state_dict'] = primary_optimizer.state_dict()
    elif args.tracepoint_save_optimizer:
        checkpoint['primary_optimizer_state_dict'] = primary_optimizer.state_dict()
        checkpoint['event_optimizer_state_dict'] = event_optimizer.state_dict()
    torch.save(checkpoint, args.checkpoint_path)


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    validate_isolated_tracepoint_args(args)
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    if args.tracepoint:
        primary_parameters, event_parameters = isolated_parameter_groups(model)
        primary_optimizer = torch.optim.AdamW(primary_parameters, lr=args.lr)
        event_optimizer = torch.optim.AdamW(
            event_parameters,
            lr=args.tracepoint_lr if args.tracepoint_lr is not None else args.lr,
        )
        event_scheduler = MultiStepLR(
            event_optimizer, args.scheduler_milestones, args.scheduler_rate
        )
    else:
        primary_optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.lr,
        )
        event_optimizer = None
        event_scheduler = None
    primary_scheduler = MultiStepLR(
        primary_optimizer, args.scheduler_milestones, args.scheduler_rate
    )
    prompt_text = get_prompt_text(label_map)
    start_epoch, best_metric = load_initial_state(
        model, primary_optimizer, event_optimizer, args, device
    )

    def evaluate(epoch, global_step, train_metrics):
        nonlocal best_metric
        metrics = test(
            model,
            testloader,
            args.visual_length,
            prompt_text,
            gt,
            gtsegments,
            gtlabels,
            device,
            # Select checkpoints with the untouched official VadCLIP evaluator.
            # Sidecar diagnostics run once after selection and never influence it.
            tracepoint=False,
            return_metrics=True,
        )
        primary_metric = resolve_selection_metric(metrics, args.selection_metric)
        metrics.update({
            'epoch': epoch + 1,
            'global_step': global_step,
            'primary_metric_name': args.selection_metric,
            'primary_metric': primary_metric,
            'train': train_metrics,
        })
        append_metrics(args.metrics_path, metrics)
        if primary_metric > best_metric:
            best_metric = primary_metric
            if args.tracepoint_save_checkpoint:
                save_best_checkpoint(
                    model,
                    primary_optimizer,
                    event_optimizer,
                    args,
                    epoch,
                    primary_metric,
                    metrics,
                )
                print('saved best checkpoint:', args.checkpoint_path)
            else:
                print('new best metric; checkpoint saving disabled')
        model.train()
        return metrics

    for e in range(start_epoch, args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
        loss_total_tracepoint = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            if args.tracepoint:
                normal_features, normal_label, normal_lengths, normal_spans = next(normal_iter)
                anomaly_features, anomaly_label, anomaly_lengths, anomaly_spans = next(anomaly_iter)
                visual_spans = torch.cat([normal_spans, anomaly_spans], dim=0).to(device)
            else:
                normal_features, normal_label, normal_lengths = next(normal_iter)
                anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)
                visual_spans = None

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2 = model(
                visual_features, None, prompt_text, feat_lengths
            )
            tracepoint_marks = text_features[1:].detach().clone() if args.tracepoint else None
            #loss1
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()
            #loss2
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()
            #loss3
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1
            loss_total3 += loss3.item()

            primary_loss = loss1 + loss2 + loss3
            primary_optimizer.zero_grad(set_to_none=True)
            primary_loss.backward()
            primary_optimizer.step()

            if args.tracepoint:
                primary_optimizer.zero_grad(set_to_none=True)
                # The inputs were detached before the primary update, so this is
                # mathematically the same sidecar objective without co-retaining
                # the full VadCLIP and event autograd graphs on UCF's 128-video batch.
                del primary_loss, text_features, logits1, logits2
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                event_optimizer.zero_grad(set_to_none=True)
                tracepoint_output = model.forward_isolated_tracepoint(
                    visual_features,
                    lengths=feat_lengths,
                    span_lengths=visual_spans,
                    mark_features=tracepoint_marks,
                )
                tracepoint_losses = tracepoint_weak_loss(
                    tracepoint_output,
                    text_labels,
                    count_cap=args.tracepoint_count_cap,
                    count_weight=args.tracepoint_count_weight,
                    duration_weight=args.tracepoint_duration_weight,
                    fragment_weight=args.tracepoint_fragment_weight,
                    align_weight=0.0,
                    hurdle_weight=args.tracepoint_hurdle_weight,
                    normal_zero_weight=args.tracepoint_normal_zero_weight,
                    normal_zero_gate_only=args.tracepoint_gate_only_zero_loss,
                )
                tracepoint_loss = tracepoint_losses['total']
                loss_total_tracepoint += tracepoint_loss.item()
                tracepoint_loss.backward()
                event_optimizer.step()
                del tracepoint_output, tracepoint_losses, tracepoint_loss
            step += i * normal_loader.batch_size * 2
            should_eval = (
                args.eval_every_steps > 0 and (i + 1) % args.eval_every_steps == 0
            ) or (
                args.eval_every_steps == 0
                and step % 1280 == 0
                and step != 0
            )
            if should_eval:
                message = ('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                if args.tracepoint:
                    message += ('| tracepoint: ', loss_total_tracepoint / (i+1))
                print(*message)
                evaluate(e, step, {
                    'loss1': loss_total1 / (i + 1),
                    'loss2': loss_total2 / (i + 1),
                    'loss3': loss_total3 / (i + 1),
                    'tracepoint_loss': loss_total_tracepoint / (i + 1) if args.tracepoint else 0.0,
                })

        primary_scheduler.step()
        if event_scheduler is not None:
            event_scheduler.step()
        completed_steps = min(len(normal_loader), len(anomaly_loader))
        epoch_metrics = {
            'loss1': loss_total1 / max(completed_steps, 1),
            'loss2': loss_total2 / max(completed_steps, 1),
            'loss3': loss_total3 / max(completed_steps, 1),
            'tracepoint_loss': loss_total_tracepoint / max(completed_steps, 1) if args.tracepoint else 0.0,
        }
        if args.eval_at_epoch_end:
            evaluate(e, completed_steps * normal_loader.batch_size * 2, epoch_metrics)

        if not args.output_dir:
            torch.save(model.state_dict(), args.current_model_path)
        if args.tracepoint_save_checkpoint and Path(args.checkpoint_path).is_file():
            payload = load_checkpoint_payload(args.checkpoint_path, map_location=device)
            model.load_state_dict(payload.state_dict, strict=not args.tracepoint)

    if args.tracepoint_save_checkpoint and not Path(args.checkpoint_path).is_file():
        fallback_metrics = {
            'epoch': max(start_epoch, args.max_epoch - 1) + 1,
            'primary_metric_name': args.selection_metric,
            'primary_metric': None,
            'note': 'no evaluation was requested; saved final state',
        }
        save_best_checkpoint(
            model,
            primary_optimizer,
            event_optimizer,
            args,
            args.max_epoch - 1,
            float('-inf'),
            fallback_metrics,
        )

    if args.tracepoint_save_checkpoint and args.model_path != args.checkpoint_path:
        payload = load_checkpoint_payload(args.checkpoint_path, map_location=device)
        torch.save(payload.state_dict, args.model_path)

    if args.tracepoint and args.tracepoint_save_checkpoint and Path(args.checkpoint_path).is_file():
        payload = load_checkpoint_payload(args.checkpoint_path, map_location=device)
        model.load_state_dict(payload.state_dict, strict=True)
        diagnostics = test(
            model,
            testloader,
            args.visual_length,
            prompt_text,
            gt,
            gtsegments,
            gtlabels,
            device,
            tracepoint=True,
            return_metrics=True,
        )
        diagnostics.update({
            'stage': 'selected_event_diagnostics',
            'selected_epoch': int(payload.metadata.get('epoch', -1)) + 1,
            'primary_metric_name': args.selection_metric,
            'primary_metric': resolve_selection_metric(diagnostics, args.selection_metric),
        })
        append_metrics(args.metrics_path, diagnostics)
        print('selected TracePoint diagnostics:', diagnostics['tracepoint'])

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)
    configure_run_paths(args, 'ucf')

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True, return_spans=args.tracepoint)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False, return_spans=args.tracepoint)
    normal_order_generator = None
    anomaly_order_generator = None
    if args.data_order_seed is not None:
        normal_order_generator = torch.Generator()
        normal_order_generator.manual_seed(args.data_order_seed)
        anomaly_order_generator = torch.Generator()
        anomaly_order_generator.manual_seed(args.data_order_seed + 1)
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=normal_order_generator,
    )
    anomaly_loader = DataLoader(
        anomaly_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=anomaly_order_generator,
    )

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map, return_spans=args.tracepoint)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    tracepoint_hidden_dim = args.tracepoint_hidden_dim if args.tracepoint_hidden_dim > 0 else None
    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device, tracepoint_enabled=args.tracepoint, tracepoint_duration_bins=args.tracepoint_duration_bins, tracepoint_hidden_dim=tracepoint_hidden_dim, tracepoint_use_history=args.tracepoint_use_history, tracepoint_hurdle_gate=args.tracepoint_hurdle_gate, tracepoint_global_hurdle_only=args.tracepoint_global_hurdle_only, tracepoint_detach_mark_features=args.tracepoint_detach_mark_features, tracepoint_init_seed=args.tracepoint_init_seed)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
