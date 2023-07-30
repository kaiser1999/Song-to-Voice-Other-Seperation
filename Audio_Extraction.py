import librosa
import numpy as np
import soundfile as sf
import os
import tensorflow as tf
from ffprobe import FFProbe

#from demucs.separate import main
#main(["--flac", "--two-stems", "vocals", "-n", "mdx_extra", "10 - 你給我聽好.flac"])

#%%
class audio_processing():
    def __init__(self, sr=44100, n_fft=2048, hop_len=512, n_mels=128, win_len=None):
        # for librosa audio processing
        self.sr = sr                        # sampling rate
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.win_len = win_len              # if None = n_ftt
        
        # Compute A-weighting values for loudness feature
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        self.a_weights = librosa.A_weighting(freqs).reshape((-1, 1))
        
    def read_audiofile(self, filename):
        audio, _ = librosa.load(filename, sr=self.sr, mono=False)
        if len(audio.shape) == 1:           # mono
            audio = np.expand_dims(audio, axis=0)
        return audio                        # n_channels x Time
        
    def write_audiofile(self, filename, audio, extension='.flac', subtype='PCM_16'):
        if len(audio.shape) != 1:
            audio = audio.transpose()
        
        if not filename.endswith(extension):
            filename = filename + extension
            
        sf.write(filename, audio, self.sr, subtype=subtype)
    
    def pad_spectrogram(self, spec, pad_size, mode="reflect"):
        pad_shape = ((0, 0), (0, 0), (0, pad_size))
        return np.pad(spec, pad_shape, mode=mode)
            
    def get_audio_from_spec(self, spec):
        return librosa.istft(spec, n_fft=self.n_fft, 
                             hop_length=self.hop_len, win_length=self.win_len)
    
    def get_magphase(self, audio):
        spec = self.get_spectrogram(audio)
        return librosa.magphase(spec)       # n_channels x Hz x Time
        
    def get_spectrogram(self, audio):
        spec = librosa.stft(y=audio, n_fft=self.n_fft, hop_length=self.hop_len,
                            win_length=self.win_len, pad_mode="reflect")
        return spec       # n_channels x Hz x Time
    
    def get_loudness(self, audio):
        magnitude, _ = self.get_magphase(audio)
        power_db = librosa.amplitude_to_db(magnitude)
        power_dba = power_db + self.a_weights
        return librosa.feature.rms(S=librosa.db_to_amplitude(power_dba), 
                                   hop_length=self.hop_len)
    
    def get_log_melspectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, 
                                                  n_mels=self.n_mels, 
                                                  hop_length=self.hop_len, 
                                                  win_length=self.win_len)
        
        return librosa.power_to_db(mel_spec)
    
    def magnitude_rescaling(self, magnitude, epsilon=1e-8):
        '''
            Parameters:
            -----------
            magnitude : 3 dimensional numpy matrix
            
            Standardized across time (strange that magnitude should be non-negative)
        '''
        mono = np.mean(magnitude, axis=0, keepdims=True)
        mu   = np.mean(mono, axis=-1, keepdims=True)
        std  = np.std(mono, axis=-1, keepdims=True)
        
        return (magnitude - mu) / (std + epsilon)
    
    def get_mfcc(self, audio):
        return librosa.feature.mfcc(y=audio, sr=self.sr, hop_length=self.hop_len)

#%%
class MusDB_HQ(audio_processing):
    '''
        mixture is seperately encoded as AAC, slightly different from sum of all sources
        Example: a spectrum with 56 time-steps
        1. dataset creation: overlap_ratio = 0.25 and frame_len = 8
        ________    ________    ________    ________    ________
              ________    ________    ________    ________
           stride size = 8*(1-0.25) = 6
        
        2. audio output: offset = 8*0.25/2 = 1:
           7            6           6           6           7
        _______-    -______-    -______-    -______-    -_______
              -______-    -______-    -______-    -______-
                  6           6           6           6
           a. num : number of frame unit being considered in concatenation
           b. _   : frame unit that is considered in concatenation
           c. -   : frame unit that is dropped
    '''
    
    extension = ".wav"
    stem = [["vocals"], ["bass", "drums", "other"]]
    
    def __init__(self, frame_sec=1, overlap_ratio=0.25, stem_lst=stem, **kwargs):
        super().__init__(**kwargs)
        self.stem_lst = stem_lst
        
        self.frame_len = int(self.sr * frame_sec / self.hop_len)
        self.frame_stride = int(self.frame_len * (1 - overlap_ratio))
        self.offset = int(self.frame_len * overlap_ratio/2)
        
        self.meta = {"track": [], "spec_Transposed shape": [], "n frame": []}
        self.input_shape = (self.frame_len, self.n_fft//2 + 1, len(stem_lst))
        self.mono_shape = (self.frame_len, self.n_fft//2 + 1, len(stem_lst)*1)
        self.stereo_shape = (self.frame_len, self.n_fft//2 + 1, len(stem_lst)*2)
        
    def seperate_n_stem(self, track_path):
        mixture_path = os.path.join(track_path, "mixture" + self.extension)
        mixture_audio = self.read_audiofile(mixture_path)
        x_magnitude, x_phase = self.get_magphase(mixture_audio)
        x_magnitude = self.magnitude_rescaling(x_magnitude)
        
        # stack on first axis for number of channels
        empty_shape = (0,) + x_magnitude.shape[1:]
        y_magnitude, y_phase = np.empty(empty_shape), np.empty(empty_shape)
        for source_lst in self.stem_lst:
            source_audio = 0
            for source in source_lst:
                source_path = os.path.join(track_path, source + self.extension)
                source_audio += self.read_audiofile(source_path)
            
            mag, pha = self.get_magphase(source_audio)
            y_magnitude = np.vstack([y_magnitude, mag])
            y_phase = np.vstack([y_phase, pha])
        
        return x_magnitude, x_phase, y_magnitude, y_phase
        
    def MusDB_HQ_generator(self, path="train"):
        path = path.decode("utf-8")
        for track_folder in os.listdir(path):
            track_path = os.path.join(path, track_folder)
            if not os.path.isdir(track_path):
                continue
            
            x_magnitude, x_phase, y_magnitude, y_phase = self.seperate_n_stem(track_path)
            
            # only pad time domain
            n_samples   = x_magnitude.shape[-1]
            pad_size    = self.frame_stride*(n_samples//self.frame_stride) + self.frame_len - n_samples
            x_magnitude = self.pad_spectrogram(x_magnitude, pad_size).T
            x_phase     = self.pad_spectrogram(x_phase, pad_size).T
            y_magnitude = self.pad_spectrogram(y_magnitude, pad_size).T
            y_phase     = self.pad_spectrogram(y_phase, pad_size).T
            
            for i in range(0, n_samples, self.frame_stride):
                x_m = tf.constant(x_magnitude[i:(i+self.frame_len)], dtype=tf.float32)
                x_t = tf.constant(x_phase[i:(i+self.frame_len)], dtype=tf.complex64)
                
                y_m = tf.constant(y_magnitude[i:(i+self.frame_len)], dtype=tf.float32)
                y_t = tf.constant(y_phase[i:(i+self.frame_len)], dtype=tf.complex64)
                yield {"magni_input": x_m, "phase_input": x_t}, {"magni_output": y_m, "phase_output": y_t}
    
    def MusDB_HQ_phase_generator(self, path="train"):
        path = path.decode("utf-8")
        for track_folder in os.listdir(path):
            track_path = os.path.join(path, track_folder)
            if not os.path.isdir(track_path):
                continue
            
            x_magnitude, x_phase, y_magnitude, y_phase = self.seperate_n_stem(track_path)
            
            # only pad time domain
            n_samples   = x_magnitude.shape[-1]
            pad_size    = self.frame_stride*(n_samples//self.frame_stride) + self.frame_len - n_samples
            x_magnitude = self.pad_spectrogram(x_magnitude, pad_size).T
            x_phase     = self.pad_spectrogram(x_phase, pad_size).T
            y_magnitude = self.pad_spectrogram(y_magnitude, pad_size).T
            y_phase     = self.pad_spectrogram(y_phase, pad_size).T
            
            for i in range(0, n_samples, self.frame_stride):
                x_m = tf.constant(x_magnitude[i:(i+self.frame_len)], dtype=tf.float32)
                x_p = tf.constant(x_phase[i:(i+self.frame_len)], dtype=tf.complex64)
                
                y_m = tf.constant(y_magnitude[i:(i+self.frame_len)], dtype=tf.float32)
                y_p = tf.constant(y_phase[i:(i+self.frame_len)], dtype=tf.complex64)
                yield {"phase_input": x_p}, {"phase_output": y_p}
    
    def MusDB_HQ_Complex_generator(self, path="train"):
        path = path.decode("utf-8")
        for track_folder in os.listdir(path):
            track_path = os.path.join(path, track_folder)
            if not os.path.isdir(track_path):
                continue
            
            x_magnitude, x_phase, y_magnitude, y_phase = self.seperate_n_stem(track_path)
            x_spectrogram = x_magnitude * x_phase
            y_spectrogram = y_magnitude * y_phase
            
            # only pad time domain
            n_samples     = x_magnitude.shape[-1]
            pad_size      = self.frame_stride*(n_samples//self.frame_stride) + self.frame_len - n_samples
            x_spectrogram = self.pad_spectrogram(x_spectrogram, pad_size).T
            y_spectrogram = self.pad_spectrogram(y_spectrogram, pad_size).T
            
            for i in range(0, n_samples, self.frame_stride):
                x_s = tf.constant(x_spectrogram[i:(i+self.frame_len)], dtype=tf.complex64)
                y_s = tf.constant(y_spectrogram[i:(i+self.frame_len)], dtype=tf.complex64)
                yield x_s, y_s
    
    def test_set_generator(self, path, file_lst=[]):
        if len(file_lst) == 0:
            walking = [file for file in os.listdir(path) if len(FFProbe(file).streams)]
        else:
            walking = [os.path.join(path, file) for file in file_lst]
        
        for track_path in walking:
            audio = self.read_audiofile(track_path)
            x_magnitude, x_phase = self.get_spectrogram(audio)
            
            n_samples   = x_magnitude.shape[-1]
            pad_size    = self.frame_stride*(n_samples//self.frame_stride) + self.frame_len - n_samples
            x_magnitude = self.pad_spectrogram(x_magnitude, pad_size).T
            x_phase     = self.pad_spectrogram(x_phase, pad_size).T
            # spec shape: Channel x Hz x Time; x shape: Time x Hz x Channel
            
            for i in range(0, n_samples, self.frame_stride):
                x_m = tf.constant(x_magnitude[i:(i+self.frame_len)], dtype=tf.float32)
                x_p = tf.constant(x_phase[i:(i+self.frame_len)], dtype=tf.complex64)
                
                yield {"magnitude": x_m, "phase": x_p}
            
            self.meta["track"].append(track_path)
            self.meta["spec_Transposed shape"].append(x_magnitude.shape)   # Time x Hz x Channel
            self.meta["n frame"].append(int(np.ceil(n_samples / self.frame_stride)))
    
    def spec_to_audio(self, spectrogram_frame, extension=".flac", subtype="PCM_16"):
        for track_name, spec_T_shape, n_frame in zip(*self.meta.values()):
            spec_frame = spectrogram_frame[:n_frame]
            del spectrogram_frame[:n_frame]
            
            n_channels = spec_T_shape[-1]
            overall_channels = n_channels * len(self.stem_lst)
            spec_Transposed = np.zeros(spec_T_shape[:-1] + (overall_channels,), dtype=np.complex64)
            for idx_frame, s_frame in enumerate(spec_frame):
                if idx_frame == 0:
                    spec_len = self.frame_len - self.offset
                    spec_Transposed[:spec_len] = s_frame[:spec_len]
                elif idx_frame == n_frame - 1:
                    spec_len = self.frame_len - self.offset
                    spec_Transposed[-spec_len:] = s_frame[-spec_len:]
                else:
                    slice_spec = slice(self.frame_stride*idx_frame + self.offset, 
                                       self.frame_stride*idx_frame + self.frame_len - self.offset)
                    spec_Transposed[slice_spec] = s_frame[self.offset:-self.offset]
            
            for idx_source, source_lst in enumerate(self.stem_lst):
                save_name = "_".join([track_name] + source_lst)
                slice_spec = slice(idx_source*n_channels, (idx_source+1)*n_channels)
                spec = spec_Transposed.T
                source_audio = self.get_audio_from_spec(spec[slice_spec])
                self.write_audiofile(save_name, source_audio, extension=extension, subtype=subtype)
      

#%%
class MusDB_HQ_OLD(audio_processing):
    '''
        mixture is seperately encoded as AAC, slightly different from sum of all sources
        Example: a spectrum with 56 time-steps
        1. dataset creation: overlap_ratio = 0.25 and frame_len = 8
        ________    ________    ________    ________    ________
              ________    ________    ________    ________
           stride size = 8*(1-0.25) = 6
        
        2. audio output: offset = 8*0.25/2 = 1:
           7            6           6           6           7
        _______-    -______-    -______-    -______-    -_______
              -______-    -______-    -______-    -______-
                  6           6           6           6
           a. num : number of frame unit being considered in concatenation
           b. _   : frame unit that is considered in concatenation
           c. -   : frame unit that is dropped
    '''
    
    extension = ".wav"
    stem = [["vocals"], ["bass", "drums", "other"]]
    
    def __init__(self, frame_sec=1, overlap_ratio=0.25, stem_lst=stem, **kwargs):
        super().__init__(**kwargs)
        self.stem_lst = stem_lst
        
        self.frame_len = int(self.sr * frame_sec / self.hop_len)
        self.frame_stride = int(self.frame_len * (1 - overlap_ratio))
        self.offset = int(self.frame_len * overlap_ratio/2)
        
        self.meta = {"track": [], "spec_Transposed shape": [], "n frame": []}
        self.input_shape = (self.frame_len, self.n_fft//2 + 1, len(stem_lst))
        self.mono_shape = (self.frame_len, self.n_fft//2 + 1, len(stem_lst)*1)
        self.stereo_shape = (self.frame_len, self.n_fft//2 + 1, len(stem_lst)*2)
        
    def seperate_n_stem(self, track_path):
        mixture_path = os.path.join(track_path, "mixture" + self.extension)
        mixture_audio = self.read_audiofile(mixture_path)
        x_magnitude, x_phase = self.get_magphase(mixture_audio)
        
        # stack on first axis for number of channels
        empty_shape = (0,) + x_magnitude.shape[1:]
        y_spectrogram = np.empty(empty_shape)
        for source_lst in self.stem_lst:
            source_audio = 0
            for source in source_lst:
                source_path = os.path.join(track_path, source + self.extension)
                source_audio += self.read_audiofile(source_path)
            
            source_spectrogram = self.get_spectrogram(source_audio)
            y_spectrogram = np.vstack([y_spectrogram, source_spectrogram])
        
        return x_magnitude, x_phase, y_spectrogram
        
    def MusDB_HQ_generator(self, path="train"):
        path = path.decode("utf-8")
        for track_folder in os.listdir(path):
            track_path = os.path.join(path, track_folder)
            if not os.path.isdir(track_path):
                continue
            
            x_magnitude, x_phase, y_spectrogram = self.seperate_n_stem(track_path)
            
            # only pad time domain
            n_samples     = x_magnitude.shape[-1]
            pad_size      = self.frame_stride*(n_samples//self.frame_stride) + self.frame_len - n_samples
            x_magnitude   = self.pad_spectrogram(x_magnitude, pad_size).T
            x_phase       = self.pad_spectrogram(x_phase, pad_size).T
            
            y_spectrogram = self.pad_spectrogram(y_spectrogram, pad_size).T
            
            for i in range(0, n_samples, self.frame_stride):
                x_m = tf.constant(x_magnitude[i:(i+self.frame_len)], dtype=tf.float32)
                x_p = tf.constant(x_phase[i:(i+self.frame_len)], dtype=tf.complex64)
                y = tf.constant(y_spectrogram[i:(i+self.frame_len)], dtype=tf.complex64)
                yield {"magnitude": x_m, "phase": x_p}, y
    
    def test_set_generator(self, path, file_lst=[]):
        if len(file_lst) == 0:
            walking = [file for file in os.listdir(path) if len(FFProbe(file).streams)]
        else:
            walking = [os.path.join(path, file) for file in file_lst]
        
        for track_path in walking:
            audio = self.read_audiofile(track_path)
            x_magnitude, x_phase = self.get_spectrogram(audio)
            
            n_samples   = x_magnitude.shape[-1]
            pad_size    = self.frame_stride*(n_samples//self.frame_stride) + self.frame_len - n_samples
            x_magnitude = self.pad_spectrogram(x_magnitude, pad_size).T
            x_phase     = self.pad_spectrogram(x_phase, pad_size).T
            # spec shape: Channel x Hz x Time; x shape: Time x Hz x Channel
            
            for i in range(0, n_samples, self.frame_stride):
                x_m = tf.constant(x_magnitude[i:(i+self.frame_len)], dtype=tf.float32)
                x_p = tf.constant(x_phase[i:(i+self.frame_len)], dtype=tf.complex64)
                
                yield {"magnitude": x_m, "phase": x_p}
            
            self.meta["track"].append(track_path)
            self.meta["spec_Transposed shape"].append(x_magnitude.shape)   # Time x Hz x Channel
            self.meta["n frame"].append(int(np.ceil(n_samples / self.frame_stride)))
    
    def spec_to_audio(self, spectrogram_frame, extension=".flac", subtype="PCM_16"):
        for track_name, spec_T_shape, n_frame in zip(*self.meta.values()):
            spec_frame = spectrogram_frame[:n_frame]
            del spectrogram_frame[:n_frame]
            
            n_channels = spec_T_shape[-1]
            overall_channels = n_channels * len(self.stem_lst)
            spec_Transposed = np.zeros(spec_T_shape[:-1] + (overall_channels,), dtype=np.complex64)
            for idx_frame, s_frame in enumerate(spec_frame):
                if idx_frame == 0:
                    spec_len = self.frame_len - self.offset
                    spec_Transposed[:spec_len] = s_frame[:spec_len]
                elif idx_frame == n_frame - 1:
                    spec_len = self.frame_len - self.offset
                    spec_Transposed[-spec_len:] = s_frame[-spec_len:]
                else:
                    slice_spec = slice(self.frame_stride*idx_frame + self.offset, 
                                       self.frame_stride*idx_frame + self.frame_len - self.offset)
                    spec_Transposed[slice_spec] = s_frame[self.offset:-self.offset]
            
            for idx_source, source_lst in enumerate(self.stem_lst):
                save_name = "_".join([track_name] + source_lst)
                slice_spec = slice(idx_source*n_channels, (idx_source+1)*n_channels)
                spec = spec_Transposed.T
                source_audio = self.get_audio_from_spec(spec[slice_spec])
                self.write_audiofile(save_name, source_audio, extension=extension, subtype=subtype)
      
