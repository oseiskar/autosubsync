Automatically synchronize SRT subtitles with audio.
Requires ffmpeg (``sudo apt-get install ffmepg``)::

  autosubsync [input movie] [input subtitles] [output subs]

  # for example
  autosubsync plan-9-from-outer-space.avi \
    plan-9-out-of-sync-subs.srt \
    plan-9-subtitles-synced.srt

See ``autosubsync --help`` for more details.
