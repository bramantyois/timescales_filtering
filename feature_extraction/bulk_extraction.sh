export OMP_NUM_THREADS=16

python save_features.py  --featureset_name BERT_all --is_bling --subject_id COL
python save_features.py  --featureset_name BERT_all --is_bling --subject_id TYE
python save_features.py  --featureset_name chinese_BERT_all --is_bling --is_chinese --subject_id COL
python save_features.py  --featureset_name chinese_BERT_all --is_bling --is_chinese --subject_id TYE
