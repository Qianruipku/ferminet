# Copyright 2018 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Writer utility classes."""

import contextlib
import os
from typing import Optional, Sequence

from absl import logging


class Writer(contextlib.AbstractContextManager):
  """Write data to CSV, as well as logging data to stdout if desired."""

  def __init__(self,
               name: str,
               schema: Sequence[str],
               directory: str = 'logs/',
               append: bool = False):
    """Initialise Writer.

    Args:
      name: file name for CSV.
      schema: sequence of keys, corresponding to each data item.
      directory: directory path to write file to.
      append: if True, append to existing file instead of overwriting.
    """
    self._schema = schema
    if not os.path.isdir(directory):
      os.mkdir(directory)
    self._filename = os.path.join(directory, name + '.csv')
    self._append = append and os.path.exists(self._filename) and os.path.getsize(self._filename) > 0

  def __enter__(self):
    mode = 'a' if self._append else 'w'
    self._file = open(self._filename, mode, encoding='UTF-8')
    # write top row of csv only if not appending or file is empty
    if not self._append:
      self._file.write(','.join(self._schema) + '\n')
    self._buffer = []
    return self

  def write(self, if_print: bool, **data):
    """Writes to buffer, only writes to file and log when flush condition is met."""
    row = [str(data.get(key, '')) for key in self._schema]
    for key in data:
      if key not in self._schema:
        raise ValueError(f'Not a recognized key for writer: {key}')
    row_str = ','.join(row) + '\n'
    self._buffer.append(row_str)

    if if_print:
      self._file.writelines(self._buffer)
      self._file.flush()
      self._buffer.clear()

  def __exit__(self, exc_type, exc_val, exc_tb):
      self._file.close()
