import unittest, os, tempfile, json

#from autosubsync import xyz
from generate_test_data import generate, set_seed
from autosubsync import synchronize

def generate_dummy_model(filename):
    DUMMY_MODEL = {
        'bias': 0.0,
        'logistic_regression': {
            'bias': -1.0,
            'coef': [
                [1.0] * 250
            ]
        }
    }
    with open(filename, 'w') as f:
        json.dump(DUMMY_MODEL, f)

class TestSync(unittest.TestCase):
    def test_sync(self):
        set_seed(0)

        temp_sound = '/tmp/sound.flac'
        temp_subs = 'tmp/subs.srt'

        tmp_dir = tempfile.mkdtemp()
        temp_sound = os.path.join(tmp_dir, 'sound.flac')
        temp_subs = os.path.join(tmp_dir, 'subs.srt')
        temp_out = os.path.join(tmp_dir, 'synced.srt')
        temp_model = os.path.join(tmp_dir, 'model.bin')

        def run_test():
            generate_dummy_model(temp_model)

            true_skew = 24/25.0
            true_shift_seconds = 4.0

            generate(temp_sound, temp_subs, true_skew, true_shift_seconds)

            # seems to work with FFMPEG without wrapping into a video file
            video_file = temp_sound

            success, quality, skew, shift = synchronize(\
                video_file, temp_subs, temp_out,
                model_file=temp_model,
                verbose=True, return_parameters=True)

            skew_error = abs(skew - true_skew)
            shift_error = abs(shift - true_shift_seconds)

            self.assertTrue(success)
            self.assertEqual(skew_error, 0.0)
            # not very accurate with short/toy data
            self.assertTrue(shift_error < 1.0)

        def clear():
            for f in [temp_sound, temp_subs, temp_out, temp_model]:
                try: os.unlink(sound_file)
                except: pass
            try: os.rmdir(tmp_dir)
            except: pass

        try: run_test()
        finally: clear()

if __name__ == '__main__':
    unittest.main()
