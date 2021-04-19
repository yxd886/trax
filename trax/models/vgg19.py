# coding=utf-8
# Copyright 2020 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""VGG19."""

from trax import layers as tl


def VGG19(dropout_keep_prob=0.5, n_output_classes=1000, mode='train',
             norm=tl.BatchNorm,
             non_linearity=tl.Relu):
  """ResNet.

  Args:
    d_hidden: Dimensionality of the first hidden layer (multiplied later).
    n_output_classes: Number of distinct output classes.
    mode: Whether we are training or evaluating or doing inference.
    norm: `Layer` used for normalization, Ex: BatchNorm or
      FilterResponseNorm.
    non_linearity: `Layer` used as a non-linearity, Ex: If norm is
      BatchNorm then this is a Relu, otherwise for FilterResponseNorm this
      should be ThresholdedLinearUnit.

  Returns:
    The list of layers comprising a ResNet model with the given parameters.
  """

  return tl.Serial(
      tl.ToFloat(),
      tl.Conv(64, (3, 3),padding="SAME"),
      tl.Conv(64, (3, 3), padding="SAME"),
      tl.MaxPool(pool_size=(3, 3), strides=(2, 2)),

      tl.Conv(128, (3, 3), padding="SAME"),
      tl.Conv(128, (3, 3), padding="SAME"),
      tl.MaxPool(pool_size=(3, 3), strides=(2, 2)),

      tl.Conv(256, (3, 3), padding="SAME"),
      tl.Conv(256, (3, 3), padding="SAME"),
      tl.Conv(256, (3, 3), padding="SAME"),
      tl.Conv(256, (3, 3), padding="SAME"),
      tl.MaxPool(pool_size=(3, 3), strides=(2, 2)),

      tl.Conv(512, (3, 3), padding="SAME"),
      tl.Conv(512, (3, 3), padding="SAME"),
      tl.Conv(512, (3, 3), padding="SAME"),
      tl.Conv(512, (3, 3), padding="SAME"),
      tl.MaxPool(pool_size=(3, 3), strides=(2, 2)),

      tl.Conv(4096, (7, 7), padding="VALID"),
      tl.Dropout(rate=dropout_keep_prob, shared_axes=None, mode=mode),

      tl.Conv(4096, (1, 1), padding="SAME"),
      tl.Dropout(rate=dropout_keep_prob, shared_axes=None, mode=mode),
      tl.Flatten(),
      tl.Dense(n_output_classes),
      tl.LogSoftmax(),
  )
