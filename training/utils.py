"""
Leftover utilities useful for Jypyter notebooks and development but not
strictly required.
"""

def read_srt_to_data_frame(fn):
    "Read SRT file to a pandas.DataFrame"

    from autosubsync import srt_io
    import pandas as pd

    rows = [list(x) for x in srt_io.read_file_tuples(fn)]
    df = pd.DataFrame(rows, columns=['seq', 'begin', 'end', 'text'])
    return df.set_index('seq')
