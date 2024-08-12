import pandas as pd
from rnnoise_wrapper import RNNoise
from pydub import AudioSegment

# path - путь до csv файла с данными о файлах
# input_path - путь до папки, где находятся изначальные аудиофайлы
# output_path - путь до папки, куда сохранять обработанные аудиофайлы 

def do_sound_to_better(path, input_path, output_path):

    train = pd.read_csv(path, header=None, names=['names', 'labels'])

    for i in train['names'].tolist():

        audio = AudioSegment.from_mp3(f'{input_path}/{i}')

        if audio.frame_rate!=48000:
            audio.set_frame_rate(48000)
        if audio.sample_width!=2:
            audio.set_sample_width(2)
        if audio.channels!=1:
            audio.set_channels(1)

        denoiser = RNNoise()

        denoised_audio = denoiser.filter(audio)
        denoised_audio.export(f'{output_path}/{i}', format="mp3")