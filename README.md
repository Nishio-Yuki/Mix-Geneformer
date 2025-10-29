# mouse-Geneformer++

工事中

編集者：伊藤啓太

# 事前学習と追加学習とファインチューニング
mouse-Geneformer++ の事前学習は，pretrain_geneformer.py で使用するモデルやタスク，エポック数，学習率等を設定して
```bash
# mosue-Genecorpus-20M dataset
./start_pretrain_geneformer.sh
```
で実行したらできます．
細胞型分類タスクや疾患状態の分類タスクへのファインチューニングは，`cell_classification.ipynb` や `in_silico_perturbation.ipynb` で行うことができます．
細胞型分類実験や in silico 摂動実験のやり方は，https://github.com/machine-perception-robotics-group/Archive-2024/tree/main/TP23001_ItoKeita/mouse-Geneformer を見てください．
