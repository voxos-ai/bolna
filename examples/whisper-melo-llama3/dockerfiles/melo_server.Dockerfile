FROM python:3.10.13-slim
WORKDIR /app

RUN apt-get update && apt-get install libgomp1 git -y
RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends ffmpeg
RUN git clone https://github.com/bolna-ai/MeloTTS
RUN pip install fastapi uvicorn torchaudio
RUN cp -a MeloTTS/. . 
RUN python -m pip cache purge
RUN pip install --no-cache-dir txtsplit torch torchaudio cached_path transformers==4.27.4 mecab-python3==1.0.5 num2words==0.5.12 unidic_lite unidic mecab-python3==1.0.5 pykakasi==2.2.1 fugashi==1.3.0 g2p_en==2.1.0 anyascii==0.3.2 jamo==0.4.1 gruut[de,es,fr]==2.2.3 g2pkk>=0.1.1 librosa==0.9.1 pydub==0.25.1 eng_to_ipa==0.0.2 inflect==7.0.0 unidecode==1.3.7 pypinyin==0.50.0 cn2an==0.5.22 jieba==0.42.1 langid==1.1.6 tqdm tensorboard==2.16.2 loguru==0.7.2
RUN python -m unidic download
EXPOSE 8000
CMD ["python3", "Server.py"]
