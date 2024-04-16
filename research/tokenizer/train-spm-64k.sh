python3 train-sentencepiece.py \
  --input="$DATA_DIR"/train.txt \
  --model-prefix="$DATA_DIR"/spm \
  --vocab-size=64000 \
  --self-test-sample-size=1000 \
  --character-coverage=0.9995 \
  --input-sentence-size=57540000 \
  --num-threads=32 \
  --user-defined-symbols-file=$DATA_DIR/user_defined_symbols.txt \
  --byte-fallback=true |& tee "$DATA_DIR"/spm_train.log

cp "$0" "$DATA_DIR"/spm_train.sh

#  --normalization-rule-tsv=$DATA_DIR/normalization_rule.tsv  -  seems to be not needed
# --self_test_sample_size=1000  -  what does it do at all???
# --character_coverage=0.9995  -  may change after metrics analysis
# --input_sentence_size=5000000  -  FIXME change to bigger value after tests
# --byte_fallback=true  -  I need to verify if this does not introduce any problem
