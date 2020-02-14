# PEL-BERT  


PAKDD2020, PEL-BERT: A Joint Model for Protocol Entity Linking  

Use Google's BERT for classification task for RFC paper（contains dataset in dir "rfc"）.   


### Folder Description:
```
BERT-NER
|____ bert                      # need git from [here](https://github.com/google-research/bert)
|____ cased_L-12_H-768_A-12	    # need download from [here](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)
|____ data		                # train data
|____ middle_data	            # middle data (label id map)
|____ output			        # output (final model, predict results)
|____ BERT_NER.py		        # mian code
|____ conlleval.pl		        # eval code
|____ run_ner.sh    		    # run model and eval result

```


### Usage:
```
run in pycharm, start file is run_classifier.py(for RFC-BERT-a/b/c , switch these by choosing line from 579 to 581); run_classifier_source.py(for BERT & Ashutosh Adhikari[2].; mask line 537 for the former, use line 537 for latter); traditional_ml_classifier.py(for Ashutosh Adhikari[1].; SVM; BPNN; CNN; Bi-GRU, switch these by choosing line from 340 to 344, change the name conformed with the above)

change the dataset from (292 and 293 for traditional_ml_classifier.py ; 255, 261, and 267 for run_classifier.py ; 238, 244, and 250 for run_classifier_source.py) when do the 10 folds validation.
```

###e.g.
```
python run_classifier.py\
    
    --do_lower_case=False \
    --do_train=True   \
    --do_eval=True   \
    --do_predict=True \
    --data_dir=./rfc   \
    --task_name="classify"  \
    --vocab_file=./cased_L-12_H-768_A-12/vocab.txt   \
    --bert_config_file=./cased_L-12_H-768_A-12/bert_config.json   \
    --output_dir=./output/   \
    --init_checkpoint=./cased_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=56   \
    --field_seq_length=10   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=3.0   

```
**Notice:** I set num_train_epochs to 18.0 by changing line 872 in run_classifier.py ; 798 in run_classifier_source.py.

**Notice:** cased model was recommened, I set in the dir cased_L-12_H-768_A-12

### Result description:
Result file is test_results.tsv, every line gives a single sample's probabilities in all categories. The maximum value is the prediction.

### reference:

[1] https://github.com/google-research/bert

### Baseline Papers

[1] Adhikari, Ashutosh, et al. "Rethinking complex neural network architectures for document classification." Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019.

[2] Adhikari, Ashutosh, Achyudh Ram, Raphael Tang, and Jimmy Lin. "DocBERT: BERT for Document Classification." arXiv preprint arXiv:1904.08398 (2019).
