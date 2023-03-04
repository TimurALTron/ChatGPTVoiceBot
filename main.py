import configparser
import os

import torch

import speech_recognition
import openai

from pydub import AudioSegment

from aiogram import Bot, Dispatcher, executor, types
from pathlib import Path
from transliterate import translit

from aiogram.types import File, InputFile

# CONFIG Init
config = configparser.ConfigParser()
config.read("config.ini")

# Init Torch
language = 'ru'
model_id = 'v3_1_ru'
sample_rate = 48000
speaker = 'eugene'
device = torch.device("cpu")
model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=language,
                          speaker=model_id)
model.to(device)

# OpenAI Init
openai.api_key = config['CLIENT']['tokenOpenAI']
model_engine = "text-davinci-003"

r = speech_recognition.Recognizer()

bot = Bot(config['CLIENT']['tokenBotTelegram'])
dp = Dispatcher(bot)


async def handle_file(file: File, file_name: str):
    Path(f"{os.getcwd()}/voices").mkdir(parents=True, exist_ok=True)
    await bot.download_file(file_path=file.file_path, destination=f"{os.getcwd()}/voices/{file_name}")


@dp.message_handler(content_types=["voice"])
async def voice_handler(message: types.Message):
    voice = await message.voice.get_file()
    await handle_file(file=voice, file_name=f"{voice.file_id}.ogg")
    given_audio = AudioSegment.from_file(f"{os.getcwd()}/voices/{voice.file_id}.ogg", format="ogg")
    given_audio.export(f"{os.getcwd()}/voices/{voice.file_id}.wav", format="wav")



    hardvard = speech_recognition.AudioFile(f"{os.getcwd()}/voices/{voice.file_id}.wav")
    with hardvard as source:
        audio = r.record(source)
        query = r.recognize_google(audio, language='ru-RU')

        await message.answer("✅ Ожидайте ответа ✅")

        prompt = query
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        finalText = translit(completion.choices[0].text, 'ru')

        await message.answer(finalText.split('\n', 1)[1])

        paths = model.save_wav(text=finalText.split('\n', 1)[1], speaker=speaker, sample_rate=sample_rate)

        await bot.send_voice(message.chat.id, open(f"{os.getcwd()}/test.wav", 'rb'))


if __name__ == "__main__":
    executor.start_polling(dp)
