"""
Module for reading and writing the SRT format. Attempts to deal with
different encodings in a pass-through fashion: read and write as binary.
"""

def read_file(input_file):
    """
    Read an SRT file to (seq, begin, end, text) tuples, where
    begin and end are float timestamps in seconds. Text is
    actually binary since we don't really care about the encoding here

    :param input_file: input file name (string)
    :return: generator of tuples ``(seq, begin, end, text)``
    """
    def parse_time(timestamp):
        hours, minutes, secs = timestamp.split(b':')
        return (int(hours)*60 + int(minutes))*60 + float(secs.replace(b',', b'.'))

    def convert_line_ending(data):
        "Convert line endings to UNIX"
        return data.replace(b'\r\n', b'\n').replace(b'\r', b'\n')

    def remove_boms(data):
        "remove UTF BOMs if present"
        BOMS = [b'\xEF\xBB\xBF', b'\xFE\xFF']
        for bom in BOMS:
            if data.startswith(bom):
                data = data[len(bom):]
        return data

    with open(input_file, 'rb') as f:
        srt_data = f.read()

    srt_data = convert_line_ending(remove_boms(srt_data))

    for line_block in srt_data.split(b'\n\n'):
        line_block = line_block.strip()
        if len(line_block) == 0: continue
        block = line_block.split(b'\n')

        seq = int(block[0])
        times = block[1]
        begin, _, end = times.partition(b' --> ')
        text = b'\n'.join(block[2:])
        yield(seq, parse_time(begin), parse_time(end), text)

class writer:
    """
    Writer for SRT files. Outputs windows line endings and no UTF BOMs
    """
    def __init__(self, file):
        """
        Create writer for an open file, which should be in binary mode

        :param file: file object opened in binary mode
        """
        self.seq = 1
        self.file = file

    def write(self, begin, end, text):
        """
        Write and SRT entry

        :param begin: begin timestamp (float seconds)
        :param end: end timestamp (float seconds)
        :param text: text (binary string)
        """
        self._write_line_ascii(self.seq)
        self._write_line_ascii(self._format_time(begin) + ' --> ' + self._format_time(end))
        self._write_line_binary(text) # can be almost any encoding
        self._write_line_ascii('') # empty line
        self.seq += 1

    def _write_line_ascii(self, thing):
        "Write something as text (must only contain ASCII characters)"
        self._write_line_binary(str(thing).encode('ascii'))

    def _write_line_binary(self, data):
        "Write binary data, for example, text in any encoding"
        data = data.rstrip().replace(b'\n', b'\r\n')
        self.file.write(data + b'\r\n')

    def _format_time(self, t_secs):
        "Convert float seconds to SRT timestamp format"
        msecs = round(t_secs*1000)
        secs = int(msecs / 1000) % 60
        mins = int(msecs / (60*1000)) % 60
        hours = int(msecs / (60*60*1000))
        msecs = msecs % 1000
        return "%02d:%02d:%02d,%03d" % (hours, mins, secs, msecs)
