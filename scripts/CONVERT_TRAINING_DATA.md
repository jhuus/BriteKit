# Convert existing training pickles

Run these from the BriteKit repository using its Hatch environment:

```bash
hatch run britekit:python scripts/convert_training_pickle.py \
  ~/code/git/HawkEars/data/training.pkl \
  ~/code/git/HawkEars/data/training

hatch run britekit:python scripts/convert_teacher_targets_pickle.py \
  ~/code/git/HawkEars/data/teacher-targets.pkl \
  ~/code/git/HawkEars/data/teacher-targets

hatch run britekit:python scripts/convert_training_frames_pickle.py \
  ~/code/git/HawkEars/data/training-frames.pkl \
  ~/code/git/HawkEars/data/training-frames
```

Only convert trusted pickle files. Each converter loads its source once, so run
these sequentially on a machine with enough available RAM for the largest pickle
plus Python/container overhead. The source files are never modified or deleted.
Conversion also needs disk space for the output (approximately the input size for
training and teacher targets). No database, audio processing, teacher inference,
or frame-label regeneration is involved.

Each output is a directory. An existing destination is refused. A sibling temporary
directory is used and renamed into place only after verification; ordinary errors
clean up the temporary directory. A killed process can leave a hidden temporary
directory, which can be removed once the process has stopped. Arrays are compared
in chunks; compressed spectrogram bytes are compared sample by sample.

Training accepts these directories in the existing configuration fields (the
field names retain `pickle` for compatibility):

```yaml
train:
  train_pickle: /home/jhuus/code/git/HawkEars/data/training
  teacher_targets_pickle: /home/jhuus/code/git/HawkEars/data/teacher-targets
  frame_label_pickle: /home/jhuus/code/git/HawkEars/data/training-frames
```

`test_pickle` also accepts a training-format store. `train --teacher-targets` and
`train --frame-labels` accept either files or store directories. Existing pickle
files still work; directories use lazy readers automatically. Preserve the audio
and training settings used for the original pickles.

Spectrograms and targets are read on demand. Validation retains an index view,
and teacher rows are aligned by ID without copying the dense arrays. Missing
teacher rows retain the hard-label fallback; absent frame IDs retain the existing
full-window fallback. Each decoded spectrogram is private writable memory for
augmentation. Mappings reopen when datasets are serialized to spawned workers.

The OS caches file pages in available RAM and reclaims them as needed. Loader
prefetch holds only a bounded number of decoded batches, but labels, sampler
indices, and ID lookup tables still need memory proportional to sample count.
Store structure is checked at startup; teacher probability values are checked
per requested row, avoiding an initial scan of the entire target payload.

## Common manifest

`manifest.json` contains `format: "britekit-indexed"`, `format_version: 1`,
`kind`, `source` (original absolute path), `count`, and `metadata`. The store is
self-contained: the source path is provenance, not a runtime dependency.
The manifest is written last. Original row, class and label ordering is preserved.
All NPY arrays contain numeric values only and can be opened with:

```python
array = np.load(path, mmap_mode="r", allow_pickle=False)
```

## Training store (`kind: "training"`)

- `specs.bin`: concatenated original compressed blobs, including existing 8/16-bit
  headers; no decompression or requantization.
- `offsets.npy`: uint64 array of N+1 byte boundaries. Sample i is the bytes in
  `[offsets[i], offsets[i+1])`.
- `label_values.npy`: int64 flattened class indexes.
- `label_offsets.npy`: uint64 array of N+1 boundaries into label_values.
- `segment_ids.npy`, `recording_ids.npy`: original integer IDs in sample order.
- Manifest metadata: original class names, codes, alternate names/codes, plus any
  additional JSON-serializable metadata from the source. Legacy files do not
  contain frontend settings; supply their original audio config when training.

Both ID lists are required. Segment IDs must be unique; recording IDs may repeat.
A reader should memory-map specs.bin read-only and copy only the requested blob
into bytes for the existing decompressor. Decoded spectrograms remain private
writable arrays for augmentation.

## Teacher-target store (`kind: "teacher-targets"`)

Accepts BriteKit version 2 teacher-target pickles.

- `segment_ids.npy`: unique IDs in original teacher-target order.
- `probabilities.npy`: original floating array, shape (N, classes).
- `frame_probabilities.npy`: optional original floating array, shape
  (N, classes, frames).
- Manifest metadata: original teacher format version, ordered class codes,
  source/checkpoint fingerprints and all other JSON-serializable metadata.

Probabilities must be finite and within [0, 1]. Original dtypes are preserved,
including float16 frame targets. A training reader must map training segment IDs
to teacher rows, rather than reorder the entire target array in memory. Absence
of a teacher ID is distinct from a stored row of zero probabilities.

## Frame-label store (`kind: "training-frames"`)

Accepts the existing dictionary mapping segment ID to a one-dimensional label
array. It does not infer or change frame timing.

- `segment_ids.npy`: dictionary keys in insertion order.
- `values.npy`: concatenated original label vectors, preserving their dtype.
- `offsets.npy`: uint64 array of N+1 boundaries into values.

Ragged vectors and empty dictionaries/vectors are supported. Values must be finite
numbers in [0, 1], and all vectors must share a dtype; mixed precision is rejected
rather than silently converted. For row i, the original labels are
`values[offsets[i]:offsets[i+1]]`. Missing segment IDs remain missing, preserving
the distinction between no supervision and an all-zero label vector.
