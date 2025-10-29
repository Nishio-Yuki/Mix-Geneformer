# Mix-Geneformer

# 事前学習と追加学習とファインチューニング
Mix-Geneformer の事前学習は，pretrain_geneformer.py で使用するモデルやタスク，エポック数，学習率等を設定して
```bash
# mosue-Genecorpus-20M dataset
./start_pretrain_geneformer.sh
```
を実行したら学習できます．途中のチェックポイントがある場合は，`start_pretrain_geneformer.sh`に`--resume_from_checkpoint`の引数を追加して，後ろに学習途中のモデルを指定してください．
細胞型分類タスクやin silico摂動実験のためのファインチューニングは，`cell_classification.ipynb` や `in_silico_perturbation.ipynb` で行うことができます．
細胞型分類実験や in silico 摂動実験のやり方は，https://github.com/machine-perception-robotics-group/Archive-2024/tree/main/TP23001_ItoKeita/mouse-Geneformer を見てください．
