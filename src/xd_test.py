import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CLIPVAD
from tracepoint import TracePointDiagnostics, infer_tracepoint_chunks, project_event_coverage
from utils.checkpoint import load_checkpoint_payload
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option

# This module exposes the project's benchmark evaluator rather than pytest cases.
__test__ = False

def test(
    model,
    testdataloader,
    maxlen,
    prompt_text,
    gt,
    gtsegments,
    gtlabels,
    device,
    tracepoint=False,
    tracepoint_fuse=False,
    return_metrics=False,
):
    if tracepoint_fuse:
        raise ValueError("TracePoint fusion is not supported by the isolated sidecar")

    model.to(device)
    model.eval()

    element_logits2_stack = []
    diagnostics = TracePointDiagnostics(model.tracepoint.duration_bins) if tracepoint else None

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = item[2]
            spans = item[3].squeeze(0) if tracepoint else None

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
                if tracepoint:
                    spans = spans.unsqueeze(0)

            chunk_count = max(1, (len_cur + maxlen - 1) // maxlen)
            visual = visual[:chunk_count].to(device)
            if tracepoint:
                spans = spans[:chunk_count].to(device)
            lengths = torch.tensor(
                [min(maxlen, len_cur - chunk * maxlen) for chunk in range(chunk_count)],
                dtype=torch.long,
            )
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            if tracepoint:
                # Official scores always come directly from VadCLIP.  The event
                # process is evaluated separately for diagnostics only.
                _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
                chunk_output = infer_tracepoint_chunks(
                    model, visual, padding_mask, lengths, spans, prompt_text
                )
                global_start_prob = chunk_output['start_prob']
                hurdle_presence_prob = chunk_output['hurdle_presence_prob']
                hurdle_global_presence_prob = chunk_output['hurdle_global_presence_prob']
                coverage_output = project_event_coverage(
                    global_start_prob,
                    model.tracepoint.duration_bins,
                    lengths=torch.tensor([len_cur], device=device),
                    span_lengths=chunk_output['span_lengths'],
                )
                tracepoint_output = {
                    'event_score': coverage_output['event_score'],
                    'coverage': coverage_output['coverage'],
                }
            else:
                _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            if tracepoint:
                video_label = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
                diagnostics.update(
                    tracepoint_output['event_score'][0, :len_cur],
                    global_start_prob,
                    is_normal=str(video_label) == 'A',
                    hurdle_presence_prob=hurdle_presence_prob,
                    hurdle_global_presence_prob=hurdle_global_presence_prob,
                )
                normal_score = 1 - tracepoint_output['event_score'].reshape(-1, 1)
                anomaly_scores = tracepoint_output['coverage'].reshape(
                    -1, tracepoint_output['coverage'].shape[-1]
                )
                element_logits2 = torch.cat([normal_score, anomaly_scores], dim=-1)[0:len_cur]
                element_logits2 = element_logits2.detach().cpu().numpy()
            else:
                element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(len(iou)):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP / len(iou)
    print('average MAP: {:.2f}'.format(averageMAP))

    if return_metrics:
        metrics = {
            'auc': float(ROC1),
            # The legacy XD protocol reports the AUC from branch 1 and AP from branch 2.
            'ap': float(AP2),
            'visual_ap': float(AP1),
            'text_auc': float(ROC2),
            'text_ap': float(AP2),
            'localization_map': {
                f'{threshold:.1f}': float(score)
                for threshold, score in zip(iou, dmap)
            },
            'average_localization_map': float(averageMAP),
            'tracepoint_fuse': False,
        }
        if diagnostics is not None:
            metrics['tracepoint'] = diagnostics.as_dict()
        return metrics

    return ROC1, AP2 ,0#, averageMAP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map, return_spans=args.tracepoint)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    tracepoint_hidden_dim = args.tracepoint_hidden_dim if args.tracepoint_hidden_dim > 0 else None
    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device, tracepoint_enabled=args.tracepoint, tracepoint_duration_bins=args.tracepoint_duration_bins, tracepoint_hidden_dim=tracepoint_hidden_dim, tracepoint_use_history=args.tracepoint_use_history, tracepoint_hurdle_gate=args.tracepoint_hurdle_gate, tracepoint_global_hurdle_only=args.tracepoint_global_hurdle_only, tracepoint_detach_mark_features=args.tracepoint_detach_mark_features)
    payload = load_checkpoint_payload(args.model_path, map_location=device)
    model.load_state_dict(payload.state_dict, strict=not args.tracepoint)

    test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device, tracepoint=args.tracepoint)
