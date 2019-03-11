"""
Module for reading and writing the SRT format. Attempts to deal with
different encodings in a pass-through fashion: read and write as binary.
"""

def read_file_tuples(input_file):
    """
    Read an SRT file to (seq, begin, end, text) tuples, where
    begin and end are float timestamps in seconds. Text is
    actually binary since we don't really care about the encoding here

    Args:
        input_file (string): input file name

    Yields:
        generator of tuples (seq, begin, end, text)
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

    def get_line_blocks(data):
        "combine invalid SRT line blocks with double line breaks"
        last_block = []
        blocks = []
        for line_block in data.split(b'\n\n'):
            line_block = line_block.strip()
            if len(line_block) == 0: continue
            block = line_block.split(b'\n')
            try: int(block[0])
            except:
                last_block.extend(block)
                continue

            blocks.append(block)
            last_block = block
        return blocks

    with open(input_file, 'rb') as f:
        srt_data = f.read()

    srt_data = convert_line_ending(remove_boms(srt_data))

    error = None
    try:
        for block in get_line_blocks(srt_data):
            seq = int(block[0])
            times = block[1]
            begin, _, end = times.partition(b' --> ')
            text = b'\n'.join(block[2:])
            yield(seq, parse_time(begin), parse_time(end), text)
    except Exception as err:
        error = RuntimeError("%s\n\n\tUnable to parse '%s'. Is it a valid SRT file?\n" \
            % (err, input_file))

    if error is not None: raise error

class SrtEntry:
    """
    SRT entry, the subtitle text visible as a whole for a certain time interval

    Attributes:
        seq (int): integer sequence of this text in the SRT file
        begin (float): begin timestamp in seconds
        end (float): end timestamp in seconds
        text (binary string): juman-readable text to be shown as a subtitle
    """
    pass

def read_file(input_file):
    """
    Read an SRT file to SrtEntry objects,

    Args:
        input_file (string): input file name

    Yields:
        generator of SrtEntry objects
    """
    for seq, begin, end, text in read_file_tuples(input_file):
        entry = SrtEntry()
        entry.seq = seq
        entry.begin = begin
        entry.end = end
        entry.text = text
        yield(entry)

def check_file(input_file):
    """
    Check that an SRT file can be read

    Args:
        input_file (string): input file name

    Throws:
        RuntimeError if the file was not valid
    """
    list(read_file(input_file))


class writer:
    """
    Writer for SRT files. Outputs windows line endings and no UTF BOMs
    """
    def __init__(self, file):
        """
        Create writer for an open file, which should be in binary mode

        Args:
            file: file object opened in binary mode
        """
        self.seq = 1
        self.file = file

    def write(self, begin, end, text):
        """
        Write an SRT entry

        Args:
            begin (float): begin timestamp
            end (float): end timestamp
            text (binary string): text
        """
        self._write_line_ascii(self.seq)
        self._write_line_ascii(self._format_time(begin) + ' --> ' + self._format_time(end))
        self._write_line_binary(text) # can be almost any encoding
        self._write_line_ascii('') # empty line
        self.seq += 1

    def write_entry(self, srt_entry):
        """
        Write an SRT entry.

        Args:
            srt_entry (SrtEntry): an object with begin, end and text attributes

        Note:
            the seq attribute, if any is ignored and automatically set to
            a running integer
        """
        self.write(srt_entry.begin, srt_entry.end, srt_entry.text)

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
