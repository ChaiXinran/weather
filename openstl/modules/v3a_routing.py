"""Object-aware routing targets for V3a preserve/motion/decay experts."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


ROUTE_IGNORE = 0
ROUTE_PRESERVE = 1
ROUTE_MOTION = 2
ROUTE_DECAY = 3


def _objects(mask):
    labels, count = ndimage.label(
        np.asarray(mask, dtype=bool),
        structure=np.ones((3, 3), dtype=np.uint8))
    result = []
    for label_index in range(1, count + 1):
        component = labels == label_index
        coordinates = np.argwhere(component)
        if coordinates.size:
            result.append({
                'mask': component,
                'area': int(component.sum()),
                'centroid': coordinates.mean(axis=0),
            })
    return result


def match_objects(pred_mask, true_mask, radius=2.0, iou_threshold=0.1,
                  area_ratio=(0.5, 2.0), iou_weight=0.5,
                  distance_weight=0.5, minimum_score=0.1,
                  ambiguity_margin=0.05):
    """One-to-one object matching with explicit ambiguous-pair rejection."""
    pred_objects, true_objects = _objects(pred_mask), _objects(true_mask)
    if not pred_objects or not true_objects:
        return [], pred_objects, true_objects, list(range(len(pred_objects))), \
            list(range(len(true_objects)))
    scores = np.full(
        (len(pred_objects), len(true_objects)), -np.inf, dtype=np.float64)
    for pred_index, pred in enumerate(pred_objects):
        for true_index, true in enumerate(true_objects):
            ratio = pred['area'] / max(true['area'], 1)
            if not area_ratio[0] < ratio < area_ratio[1]:
                continue
            intersection = np.logical_and(pred['mask'], true['mask']).sum()
            union = np.logical_or(pred['mask'], true['mask']).sum()
            iou = float(intersection / union) if union else 0.0
            distance = float(np.linalg.norm(
                pred['centroid'] - true['centroid']))
            if iou <= iou_threshold and distance >= radius:
                continue
            scores[pred_index, true_index] = (
                iou_weight * iou
                + distance_weight * np.exp(-distance / max(radius, 1e-6)))
    finite = np.isfinite(scores)
    if not finite.any():
        return [], pred_objects, true_objects, list(range(len(pred_objects))), \
            list(range(len(true_objects)))
    cost = np.where(finite, -scores, 1e6)
    pred_indices, true_indices = linear_sum_assignment(cost)
    matches, used_pred, used_true = [], set(), set()
    for pred_index, true_index in zip(pred_indices, true_indices):
        score = scores[pred_index, true_index]
        if not np.isfinite(score) or score < minimum_score:
            continue
        alternatives = []
        for values in (np.delete(scores[pred_index], true_index),
                       np.delete(scores[:, true_index], pred_index)):
            values = values[np.isfinite(values)]
            if values.size:
                alternatives.append(float(values.max()))
        alternative = max(alternatives) if alternatives else -np.inf
        matches.append({
            'pred': pred_objects[pred_index],
            'true': true_objects[true_index],
            'score': float(score),
            'ambiguous': bool(
                np.isfinite(alternative)
                and score - alternative < ambiguity_margin),
        })
        used_pred.add(pred_index)
        used_true.add(true_index)
    return (matches, pred_objects, true_objects,
            [index for index in range(len(pred_objects)) if index not in used_pred],
            [index for index in range(len(true_objects)) if index not in used_true])


def _route_threshold(pred_mask, true_mask, **match_kwargs):
    route = np.zeros(np.asarray(pred_mask).shape, dtype=np.uint8)
    matches, pred_objects, _, unmatched_pred, _ = match_objects(
        pred_mask, true_mask, **match_kwargs)
    for match in matches:
        pred, true = match['pred'], match['true']
        union = np.logical_or(pred['mask'], true['mask'])
        if match['ambiguous']:
            continue
        distance = float(np.linalg.norm(pred['centroid'] - true['centroid']))
        intersection = np.logical_and(pred['mask'], true['mask']).sum()
        iou = float(intersection / union.sum()) if union.any() else 0.0
        if distance >= 0.5 or iou < 0.8:
            route[union] = ROUTE_MOTION
        else:
            overlap = np.logical_and(pred['mask'], true['mask'])
            route[overlap] = ROUTE_PRESERVE
            route[np.logical_and(pred['mask'], ~true['mask'])] = ROUTE_DECAY
            route[np.logical_and(~pred['mask'], true['mask'])] = ROUTE_MOTION
    for index in unmatched_pred:
        route[pred_objects[index]['mask']] = ROUTE_DECAY
    return route, matches


def build_packed_routing_target(direct_rain, target_rain, radius=2.0,
                                iou_threshold=0.1, area_ratio=(0.5, 2.0),
                                iou_weight=0.5, distance_weight=0.5,
                                minimum_score=0.1,
                                ambiguity_margin=0.05):
    """Pack 16/32-mm/h route classes into one uint8 per lead/grid cell."""
    direct = np.asarray(direct_rain)
    target = np.asarray(target_rain)
    if direct.shape != target.shape or direct.ndim != 3:
        raise ValueError('direct_rain and target_rain must be [T,H,W]')
    packed = np.zeros(direct.shape, dtype=np.uint8)
    kwargs = dict(radius=radius, iou_threshold=iou_threshold,
                  area_ratio=area_ratio, iou_weight=iou_weight,
                  distance_weight=distance_weight,
                  minimum_score=minimum_score,
                  ambiguity_margin=ambiguity_margin)
    for lead in range(direct.shape[0]):
        route16, matches16 = _route_threshold(
            direct[lead] >= 16.0, target[lead] >= 16.0, **kwargs)
        route32 = np.zeros(route16.shape, dtype=np.uint8)
        for match in matches16:
            if match['ambiguous']:
                continue
            footprint = np.logical_or(
                match['pred']['mask'], match['true']['mask'])
            pred_core = (direct[lead] >= 32.0) & footprint
            true_core = (target[lead] >= 32.0) & footprint
            local_route, _ = _route_threshold(pred_core, true_core, **kwargs)
            active = local_route != ROUTE_IGNORE
            route32[active] = local_route[active]
        inherited_decay = (route16 == ROUTE_DECAY) & (direct[lead] >= 32.0)
        route32[inherited_decay] = ROUTE_DECAY
        packed[lead] = route16 | (route32 << 2)
    return packed


def decode_packed_routing_target(packed, weight16=1.0, weight32=1.5):
    """Decode packed classes to [B,T,3,H,W] soft targets and a valid mask."""
    if packed.ndim == 3:
        packed = packed.unsqueeze(0)
    if packed.ndim != 4:
        raise ValueError('packed routing target must be [B,T,H,W]')
    packed = packed.long()
    class16, class32 = packed & 3, (packed >> 2) & 3
    valid16, valid32 = class16 > 0, class32 > 0
    one16 = F.one_hot((class16 - 1).clamp_min(0), 3).float()
    one32 = F.one_hot((class32 - 1).clamp_min(0), 3).float()
    one16 = one16 * valid16.unsqueeze(-1)
    one32 = one32 * valid32.unsqueeze(-1)
    denominator = (float(weight16) * valid16.float()
                   + float(weight32) * valid32.float())
    probability = (float(weight16) * one16 + float(weight32) * one32)
    probability = probability / denominator.clamp_min(1.0).unsqueeze(-1)
    return (probability.permute(0, 1, 4, 2, 3).contiguous(),
            denominator > 0)
