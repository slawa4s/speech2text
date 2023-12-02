import io

from telegram import File, ForceReply
import torch
from scipy.signal import decimate
from telegram.ext import Application, MessageHandler, filters, CommandHandler, CallbackContext, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from dotenv import load_dotenv
from transformers import Wav2Vec2Processor, SpeechEncoderDecoderModel
import soundfile as sf
import logging
import os

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

FEEDBACK_PROMPT = "Please provide your feedback."
TRANSLATION_ACCURACY_QUESTION = "Is this the correct translation?"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARNING
)
logging.getLogger("httpx")
logger = logging.getLogger(__name__)

MODEL_ID = "bond005/wav2vec2-mbart50-ru"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = SpeechEncoderDecoderModel.from_pretrained(MODEL_ID)

savedData = []
unique_id_to_id = {}


async def saveData(voice_file_id: str, context: CallbackContext, text: str):
    voice_file_for_download: File = await context.bot.get_file(voice_file_id)
    voice_byte: bytearray = await voice_file_for_download.download_as_bytearray()
    voice_data, sample_rate = get_correct_voice_data_and_rate(voice_byte)

    savedData.append({"audio": voice_data, "text": text, "sample_rate": sample_rate})


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

    processed = processor(voice_data, sampling_rate=sample_rate, return_tensors="pt", padding='longest')
    num_processes = max(1, os.cpu_count())

    with torch.no_grad():
        predicted_ids = model.generate(**processed)

    predicted_sentence: str = processor.batch_decode(
        predicted_ids,
        num_processes=num_processes,
        skip_special_tokens=True
    )[0]

    logger.log(logging.INFO, predicted_sentence)

    unique_id_to_id[update.message.voice.file_unique_id] = update.message.voice.file_id

    keyboard = [
        [
            InlineKeyboardButton("👍", callback_data=f"accuracy_pos_{update.message.voice.file_unique_id}"),
            InlineKeyboardButton("👎", callback_data=f"accuracy_neg_{update.message.voice.file_unique_id}"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(text=f"{predicted_sentence}\n\n{TRANSLATION_ACCURACY_QUESTION}",
                                    reply_markup=reply_markup)


# The rest of the code will remain same

async def accuracy_feedback_button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    predicted_sentence = '\n'.join(query.message.text.split('\n')[:-2])

    feedback_on_id = query.data.split("_")[2]

    if query.data.startswith('accuracy_neg'):
        reply = 'Could you provide the accurate translation?'
        keyboard = [
            [
                InlineKeyboardButton("👍", callback_data=f"feedback_pos_{feedback_on_id}"),
                InlineKeyboardButton("👎", callback_data=f"feedback_neg_{feedback_on_id}"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(text=f'{predicted_sentence}\n\n{reply}', reply_markup=reply_markup)
    elif query.data.startswith('accuracy_pos'):
        reply = 'Thank you for your evaluation!'
        await saveData(unique_id_to_id[feedback_on_id], context, predicted_sentence)
        await query.edit_message_text(text=f"{predicted_sentence}\n\n{reply}")


async def handle_reply(update: Update, context: CallbackContext):
    message_id = unique_id_to_id[update.message.reply_to_message.text]
    if message_id is None:
        return
    await update.message.reply_text(text='Thank you for reply')
    await update.message.reply_to_message.delete()
    await saveData(message_id, context, update.message.text)


async def ask_for_correct_translation_button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    predicted_sentence = '\n'.join(query.message.text.split('\n')[:-2])

    if query.data.startswith('feedback_neg'):
        await query.edit_message_text(
            text=f"{predicted_sentence}\n\nPlease ignore the previous suggestion and try a different translation.",
            reply_markup=None)
    elif query.data.startswith('feedback_pos'):
        feedback_on = query.data.split("_")[2]
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=feedback_on,
            reply_markup=ForceReply(input_field_placeholder=predicted_sentence))
        await query.delete_message()


def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))

    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    application.add_handler(MessageHandler(filters.REPLY, handle_reply))

    application.add_handler(CallbackQueryHandler(accuracy_feedback_button, pattern='^accuracy_.*'))
    application.add_handler(CallbackQueryHandler(ask_for_correct_translation_button, pattern='^feedback_.*'))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


# The rest of the code will remain same

if __name__ == '__main__':
    main()
