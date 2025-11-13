Pulling down the folder and installing dependencies 


==============================================================================
requirements 

Python 3.10 or later
Git
A GPU is recommended but the code will also run on CPU (it will just be slower)
================================================================================
Clone respository 

git clone https://github.com/Joshoneill6649/smishing-nlp-josh-oneill.git
cd smishing-nlp-josh-oneill
================================================================================
create and activate virtual enviroment 

python -m venv smishing_msc_win
smishing_msc_win\Scripts\activate
===============================================================================
Install dependencies 

pip install --upgrade pip
pip install -r requirements.txt
=================================================================================


If PyTorch complains about the Python version, install Python 3.11 and repeat the steps above.

Project run order for scripts 

1)normalize_merge_english.py

2)split.py

3)run_baselines.py

4)run_bert.py

5)run_bert.py

6)run_bert_preproc256_v1.py

7)run_distilbert.py

8)run_distilbert_preproc256_v1.py

9)baseline_model_comparisons.py

10)emotion_fusion_model_1.py

11)train_emotion_tfidf_fusion_lsvc.py

12)overall_evaluation.py


run command line prediction on a message : predict_message.py ( needs trained models to work )

user_profiles.py and text_preprocess.py are helpers

