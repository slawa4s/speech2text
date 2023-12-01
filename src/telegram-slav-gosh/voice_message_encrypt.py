import io

from telegram import Update, File
from scipy.signal import decimate
from telegram.ext import CommandHandler, Application, MessageHandler, filters, CallbackContext
from dotenv import load_dotenv
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import soundfile as sf
import logging
import os

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARNING
)
logging.getLogger("httpx")
logger = logging.getLogger(__name__)

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Hi! Send voice message to bot")


def get_correct_voice_data_and_rate(voice_byte: bytearray, target_sample_rate=16000):
    buffer = io.BytesIO(voice_byte)
    buffer.seek(0)
    voice_data_raw, sample_rate = sf.read(buffer)
    reduced_data = decimate(voice_data_raw, sample_rate // target_sample_rate)
    return reduced_data, target_sample_rate


async def handle_voice_message(update: Update, context: CallbackContext):
    voice_file_for_download: File = await context.bot.get_file(update.message.voice.file_id)
    voice_byte: bytearray = await voice_file_for_download.download_as_bytearray()
    voice_data, sample_rate = get_correct_voice_data_and_rate(voice_byte)
    transcription_tensor = processor(voice_data, sampling_rate=sample_rate, return_tensors="pt")
    generated_ids = model.generate(
        transcription_tensor["input_features"],
        attention_mask=transcription_tensor["attention_mask"],
    )
    await update.message.reply_text(processor.batch_decode(generated_ids, skip_special_tokens=True))


def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
