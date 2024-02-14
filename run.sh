# 
export CUDA_VISIBLE_DEVICES=0
file=download_pdbs.py
file=parse_entries.py
file=create_examples.py
file=k-fold-CV.py
nohup python $file \
    > $file.log.2 2>&1 &