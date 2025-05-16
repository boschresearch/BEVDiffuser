# This source code is from BEVFormer
#   (https://github.com/fundamentalvision/BEVFormer)
# Copyright (c) 2022 BEVFormer authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from mmcv.utils.registry import Registry, build_from_cfg

SAMPLER = Registry('sampler')


def build_sampler(cfg, default_args):
    return build_from_cfg(cfg, SAMPLER, default_args)
