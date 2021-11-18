# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from bisect import bisect_right
import time

import torch
import numpy as np

from rekognition_online_action_detection.evaluation import compute_result

from ..base_inferences.perframe_det_batch_inference import do_perframe_det_batch_inference


from ..engines import INFERENCES as registry
@registry.register('LSTR')
def do_lstr_batch_inference(cfg,
                            model,
                            device,
                            logger):
    if cfg.MODEL.LSTR.INFERENCE_MODE == 'stream':
        do_lstr_stream_inference(cfg,
                                 model,
                                 device,
                                 logger)
    else:
        do_perframe_det_batch_inference(cfg,
                                        model,
                                        device,
                                        logger)


def do_lstr_stream_inference(cfg, model, device, logger):
    # Setup model to test mode
    model.eval()

    # Collect scores and targets
    pred_scores = []
    gt_targets = []

    def to_device(x, dtype=np.float32):
        return torch.as_tensor(x.astype(dtype)).unsqueeze(0).to(device)

    long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
    long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
    long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
    work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
    work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
    work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES

    if len(cfg.DATA.TEST_SESSION_SET) != 1:
        raise RuntimeError('Only support testing one video each time for stream inference, will fix later')

    with torch.no_grad():
        for session_idx, session in enumerate(cfg.DATA.TEST_SESSION_SET):
            visual_inputs = np.load(osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.VISUAL_FEATURE, session + '.npy'), mmap_mode='r')
            motion_inputs = np.load(osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.MOTION_FEATURE, session + '.npy'), mmap_mode='r')
            target = np.load(osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.TARGET_PERFRAME, session + '.npy'))

            start_time = time.time()

            for work_start, work_end in zip(range(0, target.shape[0] + 1),
                                            range(work_memory_length, target.shape[0] + 1)):
                # Get target
                target = target[::work_memory_sample_rate]

                # Get work memory
                work_indices = np.arange(work_start, work_end).clip(0)
                work_indices = work_indices[::work_memory_sample_rate]
                work_visual_inputs = to_device(visual_inputs[work_indices])
                work_motion_inputs = to_device(motion_inputs[work_indices])

                # Get long memory
                long_end = work_start - 1
                if long_end == -1:
                    long_indices = [0 for _ in range(long_memory_num_samples)]
                    long_visual_inputs = to_device(visual_inputs[long_indices])
                    long_motion_inputs = to_device(motion_inputs[long_indices])
                elif long_end % long_memory_sample_rate == 0:
                    long_indices = long_indices[1:] + [long_end]
                    long_visual_inputs = to_device(visual_inputs[[long_end]])
                    long_motion_inputs = to_device(motion_inputs[[long_end]])
                else:
                    long_visual_inputs = None
                    long_motion_inputs = None

                # Get memory key padding mask
                memory_key_padding_mask = np.zeros(len(long_indices))
                last_zero = bisect_right(long_indices, 0) - 1
                if last_zero > 0:
                    memory_key_padding_mask[:last_zero] = float('-inf')
                memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32)).unsqueeze(0).to(device)

                score = model.stream_inference(
                    long_visual_inputs,
                    long_motion_inputs,
                    work_visual_inputs,
                    work_motion_inputs,
                    memory_key_padding_mask)[0]
                score = score.softmax(dim=-1).cpu().numpy()

                if work_start == 0:
                    gt_targets.extend(list(target[:work_end]))
                    pred_scores.extend(list(score))
                else:
                    gt_targets.append(list(target[work_end - 1]))
                    pred_scores.append(list(score[-1]))

            result = compute_result['perframe'](
                cfg,
                gt_targets,
                pred_scores,
            )
            logger.info('mAP of video {}: {:.5f}'.format(session, result['mean_AP']))

            end_time = time.time()
            logger.info('Running time: {:.3f} seconds'.format(end_time - start_time))
