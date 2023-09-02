how_to_cut_audio_markdown = '''
  - `start-end` (e.g. 0.5-2.5) — *removes* the specified range (in seconds),
    - `start-` or just `start` — *removes* from the specified time to the end
    - `-end` -- **removes** from the start to the specified time
  - `start-end+start-end` — *removes* the range before `+` and *inserts* the range after `+` instead. Note that, unlike the remove range, the insert range must be fully specified.
  - `start-end+sample_id@start-end` — same as above, but the insert range is taken from the specified sample, even if it is in another project (mix and match!)
  - `+sample_id@start-end` — same as above, but the range from the other sample is added *to the end* of the current sample
  - `+start-end` — *keeps* just the specified range and removes everything else.
  You can combine several of the above by using a comma (`,`). **KEEP IN MIND** that in this case the ranges are applied sequentially, so the order matters. For example, `0-1,2-3` will first remove 0-1s, and will then remove 2-3s FROM THE ALREADY MODIFIED SAMPLE, so it will actually remove ranges 0-1s and *3-4s* from the original sample. This is intentional, as it allows for a step-by-step approach to editing the audio, where you add new specifiers as you listen to the result of the previous ones.
'''