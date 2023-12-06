# speech2text

- Viacheslav Suvorov
- Georgii Zorabov

Fine-tuned whisper model for ASR of Russian language.

**[Model](https://arxiv.org/pdf/2212.04356.pdf)**

**[Dataset](https://paperswithcode.com/dataset/golos)**

The training process is in [train.py](https://github.com/slawa4s/speech2text/blob/main/src/train.ipynb)
Bot sources are in [message_recognizer-bot/src](https://github.com/slawa4s/speech2text/tree/main/message_recognizer-bot/src)
To execute [voice_message_encrypt.py](https://github.com/slawa4s/speech2text/blob/main/message_recognizer-bot/src/voice_message_encrypt.py) provide proper

```bash
export TELEGRAM_TOKEN=...
export REGION=...
export AWS_KEY_ID=...
export AWS_SECRET_KEY=...
export BUCKET_NAME=...
```
