Pulling down the folder and installing dependencies 

Python 3.10 or later Git A GPU is recommended but the code will also run on CPU

Clone respository
Git clone https://github.com/Joshoneill6649/smishing-nlp-josh-oneill.git 
cd smishing-nlp-josh-oneill

Create and activate virtual enviroment
python -m venv smishing_msc_win smishing_msc_win\Scripts\activate

Install dependencies
pip install --upgrade pip pip install -r requirements.txt


Install dependencies
pip install -r requirements.txt


If PyTorch complains about the Python version, install Python 3.11 and repeat the steps above.

Project run order for scripts 


File running order
1 scripts/normalize_merge_english.py
2	scripts/split.py
3	scripts/run_baselines.py
4	scripts/run_bert.py
5	scripts/run_bert_preproc256_v1.py
6	scripts/run_distilbert.py
7	scripts/run_distilbert_preproc256_v1.py
8	scripts/baseline_model_comparisons.py
9	scripts/emotion_fusion_model_1.py
10 scripts/ train_emotion_tfidf_fusion_lsvc.py
11 scripts/overall_evaluation.py
12 scripts/user_profiles.py

13	python scripts/predict_message.py - command line interactive message prediction ( needs trained models and files in order to run)

scripts/text_preprocess.py – used by files throughout to clean data

