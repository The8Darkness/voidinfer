# CLI examples

Committed, offline `--messages` inputs for the `voidinfer` CLI (and, through
[`send_to_serve.py`](send_to_serve.py), for a running `voidinfer-serve`). [`manifest.json`](manifest.json)
lists every case with its expected observation, prompt-token count and a recommended budget. Run from
the repository root: media paths in the JSON files are repository-relative.

```powershell
$cli    = '.\build\apps\voidinfer.exe'
$model  = @('--target', 'C:\models\Qwen3.8-27B-exl3-6bpw-H6-V6', '--draft', 'C:\models\Qwen3.8-27B-DFlash2-EXL3-5bpw')
$vision = @('--vision', 'C:\models\Qwen3.8-27B-bf16')

& $cli @model --messages examples\cli\messages\text_smoke_zh.json --no-thinking --greedy --max-new 8
```

Expected stdout is exactly `42`. `--no-thinking --greedy` is the comparison mode for the short cases.

## Cases

| Case | Flags | Expected observation |
|---|---|---|
| `text_smoke_zh` | | `42` |
| `text_chat_history` | | `Cedar\|2041-09-17\|37` |
| `text_code_review` | | empty input divides by zero; explicit empty-input handling |
| `image_chart` | `--vision` | `NIFER VISION 731`; three red circles; blue square on the left |
| `image_natural` | `--vision` | mailbox `24`; sun on the right |
| `video_temporal` | `--vision` | red circle moves; green `3`; square still visible at `3`; ending `9` |
| `multi_image_compare` | `--vision` | two circles, three circles, a new yellow star |
| `mixed_image_video` | `--vision` | `NIFER-9` |
| `mixed_multiturn` | `--vision` | `24-9` |
| `thinking_logic_grid` | thinking, `--max-new 8192` | the unique grid solution (facts in the manifest) |
| `thinking_multimodal_checksum` | thinking, `--vision` | `CHECKSUM=2238` |
| `long_niah_8k` / `64k` / `128k` / `256k` | `--max-context 262144` | `ORCHID=493817; COLOR=COBALT` |
| `long_decode_aime26_01` / `15` / `30` | thinking, large `--max-new` | boxed `277` / `83` / `393` |
| `scenario_*` (code, story, translation, structured) | | structural requirements in the manifest |

The `scenario_*` prompts are the twelve scenario prompts used in the
[engine comparison](../../docs/ninfer-comparison-2026-10-12.md). Their prompt-token counts in the
manifest were measured with the Qwen3.6 frontend; Qwen3.8 carries its own tokenizer, so check the
prepared token count in the CLI's stderr when it matters.

## Media

All PNG and MP4 files are project-authored deterministic scenes; the video is five seconds at 8 FPS
(forty H.264 frames). No case depends on a network URL.
