# CLI contract

All eight metric entry points use the same top-level contract:

```text
uv run <metric> (--vbench | --audit | --both)
                (--video FILE | --video-dir DIR)
                [--output DIR] [--gpu [IDS]]
                [--metadata FILE] [--seed INT]
```

`--vbench`, `--audit`, and `--both` are required and mutually exclusive;
`--video` and `--video-dir` are also mutually exclusive. A single video may
have any filename. For a directory input, only direct children named
`video_<digits>.mp4` are accepted, with at least three digits (for example
`video_003.mp4` and `video_1000.mp4`). They are ordered by the numeric part,
not by lexical filename order. Invalid names, duplicate numeric indices, or a
directory with no matching videos are errors before model construction.

The default output root is the workspace's `output/` directory (resolved from
the repository, not from the caller's current working directory).
`--output ./artifacts/vbench-runs` selects another output directory. Every
invocation gets one run ID. With `--both`, the `vbench` and `audit` records are
separate backend directories with the same run ID, for example:

```text
artifacts/vbench-runs/dynamic-degree/
├── vbench/20260914T120000Z/
└── audit/20260914T120000Z/
```

The input path and metadata identity are recorded; videos are not renamed or
copied by the CLI.

## GPU selection and scheduling

`--gpu` with no value selects visible logical device `0`. `--gpu 0,2` means
logical devices 0 and 2 in the process's visible CUDA namespace. If the
process was started with `CUDA_VISIBLE_DEVICES=2,4,6`, those logical IDs map
to physical devices 2 and 6. The CLI never silently remaps an invalid ID to
CPU or to another GPU.

For multiple videos, work is assigned round-robin (`videos[k::len(gpu_ids)]`)
and one worker per selected GPU is started concurrently. Results retain the
worker's logical GPU ID and are merged in deterministic input order.

## Metadata

Metadata is a JSON object containing a `videos` array. Each item must name a
video and may carry a prompt plus dimension-specific fields. This is the
smallest useful shape for a mixed directory:

```json
{
  "videos": [
    {
      "video": "video_003.mp4",
      "prompt": "a person walks beside a red car",
      "dimension_metadata": {
        "scene": "street"
      }
    }
  ]
}
```

The required minimum differs by dimension:

| Dimension | Minimum metadata |
|---|---|
| Dynamic Degree | no dimension field for the official path; audit may use `motion_target` or a prompt |
| Motion Smoothness | no dimension field |
| Subject Consistency | a video is sufficient for the visual scorer; a prompt is retained when supplied |
| Scene | `dimension_metadata.scene` (or the equivalent supported scene label) |
| Human Action | official filename target, or audit `dimension_metadata.target_action` / an exact target prompt |
| Spatial Relationship | `dimension_metadata.spatial_relationship` containing `object_a`, `object_b`, and `relationship` |
| Overall Consistency | a non-empty `prompt`; optional semantic conditions may be nested in `dimension_metadata` |
| Multiple Objects | `dimension_metadata.target_objects` (or the supported object collection) |

Human Action's `--vbench` backend intentionally derives its target from the
original filename. A generic name such as `video_000.mp4` does not encode an
action, so its official result may be low or unsupported; metadata is not used
to silently replace that official filename semantics, and the CLI does not
rename the video.

For example, a spatial record is:

```json
{
  "video": "video_003.mp4",
  "prompt": "a ball is left of a box",
  "dimension_metadata": {
    "spatial_relationship": {
      "object_a": "ball",
      "object_b": "box",
      "relationship": "left of"
    }
  }
}
```

Scene audit supports `--audit-variant global` and
`--audit-variant environment_grounded`; the variant is recorded as an audit
configuration and does not change the official VBench backend.

Model-dependent commands require their model extras, external source/builds,
and locally configured checkpoints. Third-party model constructors can
download a default pretrained model when invoked (for example an OpenCLIP
constructor with `--pretrained`); this workspace run did not download weights.
For controlled runs, pre-stage local checkpoints and pass the metric's local
checkpoint option/path. Installing Python model dependencies is not the same
as installing model weights.
