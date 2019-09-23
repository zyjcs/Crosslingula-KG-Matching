# About this Code
This is the code for our ACL paper entitled **_Cross-lingual Knowledge Graph Alignment via Graph Matching Neural Network_**.

# Env Setting
Python 3.5 (**important!**)\
Tensorflow 1.8.0\
scipy\
tqdm\
argparse\
codecs

# How To Run The Codes
To train your model, you need:

(1) Generate the training data by using the following command under DBP15K dataset: (take zh_en as an example)
    
    python3 preprocessor.py zh_en train 20  # gen the training examples
    python3 preprocessor.py zh_en test 1000 # gen the test examples
    python3 preprocessor.py zh_en dev  1000  # gen the dev examples
    
    Note:
    For the first time, it may take almost 3-4 hours to generate the candiate file.
    You may also choose to directly download these files from https://drive.google.com/open?id=1dYJtj1_J4nYJdrDY95ucGLCuZXDXI7PL and directly use them to train the model.
    
(2) Train & Test the model: (take zh_en as an example)
  
    python3 run_model.py train zh_en zh_en_model -epochs=10 -use_pretrained_embedding
    python3 run_model.py test zh_en zh_en_model -use_pretrained_embedding
    
# How To Cite The Codes
Please cite our work if you like or are using our codes for your projects!

Kun Xu, Mo Yu, Yansong Feng, Yan Song, Zhiguo Wang and Dong Yu,
"Cross-lingual Knowledge Graph Alignment via Graph Matching Neural Network", arXiv preprint arXiv:1905.11605.
 
@article{xu2019graphmatching, 
title={Cross-lingual Knowledge Graph Alignment via Graph Matching Neural Network}, 
author={Xu, Kun and Wang, Liwei and Yu, Mo and Feng, Yansong and Song, Yan and Wang, Zhiguo and Yu, Dong}, 
year={2019} 
}  


# ==============================================================
## 以下是本人复现的结果（zengyujian）2019-9-21
* python3 run_model.py train zh_en zh_en_model -epochs=10 -use_pretrained_embedding
loading pretrained embedding ...
load 45680 pre-trained word embeddings from Glove
reading training data into the mem ...
100%|██████████████████████████████████████████████████| 19388/19388 [00:00<00:00, 62060.02it/s]
100%|██████████████████████████████████████████████████| 19572/19572 [00:00<00:00, 45757.04it/s]
100%|█████████████████████████████████████████████████| 90000/90000 [00:00<00:00, 121158.69it/s]
reading development data into the mem ...
100%|██████████████████████████████████████████████████| 19388/19388 [00:00<00:00, 61492.18it/s]
100%|██████████████████████████████████████████████████| 19572/19572 [00:00<00:00, 38951.28it/s]
100%|███████████████████████████████████████████████| 210000/210000 [00:00<00:00, 406802.85it/s]
writing word-idx mapping ...

Instructions for updating:
Use `tf.global_variables_initializer` instead.
100%|███████████████████████████████████████████████████████| 2813/2813 [52:40<00:00,  1.12s/it]
evaluating the model on the dev data ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10500/10500 [09:30<00:00, 18.40it/s]
Already saved model to saved_model/zh_en_model_win1_1_win2_2_node1dim_100_node2dim_100_word_embedding_dim_300_layer1_1_layer2_1_first_gcn_type_mean_pooling_second_gcn_type_mean_pooling_cosine_MP_dim_10_drop_out_0.0_use_Glove_True_pm_graph_level_sample_size_per_layer_1/model-0
writing prediction file...
-----------------------
time:2019-09-20T17:43:04.498003
Epoch 1
Loss on train:226.4307966362685
acc @1 on Dev:0.6396190476190476
acc @10 on Dev:0.7127619047619047
best acc @1 on Dev:0.6396190476190476
-----------------------
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2813/2813 [51:16<00:00,  1.09s/it]
evaluating the model on the dev data ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10500/10500 [09:19<00:00, 18.78it/s]
-----------------------
time:2019-09-20T18:43:40.571299
Epoch 2
Loss on train:106.72306765305257
acc @1 on Dev:0.6390476190476191
acc @10 on Dev:0.7126666666666667
best acc @1 on Dev:0.6396190476190476
-----------------------
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2813/2813 [51:33<00:00,  1.10s/it]
evaluating the model on the dev data ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10500/10500 [09:31<00:00, 18.37it/s]
-----------------------
time:2019-09-20T19:44:45.767623
Epoch 3
Loss on train:60.20048695024457
acc @1 on Dev:0.6376190476190476
acc @10 on Dev:0.710952380952381
best acc @1 on Dev:0.6396190476190476
-----------------------
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2813/2813 [52:11<00:00,  1.11s/it]
evaluating the model on the dev data ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10500/10500 [09:29<00:00, 18.45it/s]
-----------------------
time:2019-09-20T20:46:26.985512
Epoch 4
Loss on train:42.51080206113568
acc @1 on Dev:0.636
acc @10 on Dev:0.7108571428571429
best acc @1 on Dev:0.6396190476190476
-----------------------
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2813/2813 [52:02<00:00,  1.11s/it]
evaluating the model on the dev data ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10500/10500 [09:29<00:00, 18.44it/s]
-----------------------
time:2019-09-20T21:47:58.843034
Epoch 5
Loss on train:32.922637851032505
acc @1 on Dev:0.6315238095238095
acc @10 on Dev:0.7100952380952381
best acc @1 on Dev:0.6396190476190476
-----------------------
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2813/2813 [51:38<00:00,  1.10s/it]
evaluating the model on the dev data ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10500/10500 [09:31<00:00, 18.36it/s]
-----------------------
time:2019-09-20T22:49:09.797135
Epoch 6
Loss on train:27.051731771520508
acc @1 on Dev:0.555047619047619
acc @10 on Dev:0.6430476190476191
best acc @1 on Dev:0.6396190476190476
-----------------------
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2813/2813 [51:53<00:00,  1.11s/it]
evaluating the model on the dev data ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10500/10500 [09:32<00:00, 18.33it/s]
-----------------------
time:2019-09-20T23:50:35.769508
Epoch 7
Loss on train:21.96949219837643
acc @1 on Dev:0.6130476190476191
acc @10 on Dev:0.6917142857142857
best acc @1 on Dev:0.6396190476190476
-----------------------
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2813/2813 [52:22<00:00,  1.12s/it]
evaluating the model on the dev data ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10500/10500 [09:33<00:00, 18.32it/s]
-----------------------
time:2019-09-21T00:52:31.894280
Epoch 8
Loss on train:22.213771471683664
acc @1 on Dev:0.6210476190476191
acc @10 on Dev:0.7066666666666667
best acc @1 on Dev:0.6396190476190476
-----------------------
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2813/2813 [51:33<00:00,  1.10s/it]
evaluating the model on the dev data ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10500/10500 [09:32<00:00, 18.33it/s]
-----------------------
time:2019-09-21T01:53:38.712480
Epoch 9
Loss on train:19.19073239621629
acc @1 on Dev:0.6241904761904762
acc @10 on Dev:0.7068571428571429
best acc @1 on Dev:0.6396190476190476
-----------------------
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2813/2813 [51:32<00:00,  1.10s/it]
evaluating the model on the dev data ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10500/10500 [09:33<00:00, 18.31it/s]
-----------------------
time:2019-09-21T02:54:44.647368
Epoch 10
Loss on train:15.01722686937574
acc @1 on Dev:0.6197142857142857
acc @10 on Dev:0.6947619047619048
best acc @1 on Dev:0.6396190476190476

## 训练时间大约十小时 （P100-GPU）

## TEST  测试时间九个半小时
*  python3 run_model.py test zh_en zh_en_model -use_pretrained_embedding

loading pretrained embedding ...
load 45680 pre-trained word embeddings from Glove
reading word idx mapping from file ...
reading training data into the mem ...
100%|██████████████████████████████████████████████████| 19388/19388 [00:00<00:00, 45198.33it/s]
100%|██████████████████████████████████████████████████| 19572/19572 [00:00<00:00, 37094.39it/s]
100%|███████████████████████████████████████████| 10500000/10500000 [00:21<00:00, 479779.29it/s]
2019-09-21 11:52:19.284507: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
WARNING:tensorflow:From /home/zengyj/Crosslingula-KG-Matching/graph_match_utils.py:146: calling reduce_max (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
WARNING:tensorflow:From /home/zengyj/Crosslingula-KG-Matching/graph_match_utils.py:147: calling reduce_mean (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
WARNING:tensorflow:From /home/zengyj/Crosslingula-KG-Matching/model.py:392: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
Instructions for updating:

Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See @{tf.nn.softmax_cross_entropy_with_logits_v2}.

100%|█████████████████████████████████████████████████| 105000/105000 [9:30:05<00:00,  3.07it/s]
-----------------------
# acc @1 on Test:0.6451428571428571
# acc @10 on Test:0.758095238095238
-----------------------
writing prediction file...

