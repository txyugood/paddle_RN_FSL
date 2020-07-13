# paddle_RN_FSL
implementation of Relation Network base on PaddlePaddle
本项目为论文《Learning to Compare- Relation Network for Few-Shot Learning》中的 FSL算法的复现。基于 Paddle实现了Relation Net 小样本端到端的训练。</br>
文章地址https://www.jianshu.com/p/ebb40f995854</br>
AI Studio 项目地址：https://aistudio.baidu.com/aistudio/projectdetail/622201</br>
数据集下载地址：https://pan.baidu.com/s/1I8fYw8dhDUFdSdqt3_Jd6Q 密码: n6u2</br>

使用说明：</br>
运行 one shot learing方法如下。</br>

    python -u train.py --c_way 5 --k_shot 1 --query_num_per_class 15 \
    --episode 100000 --test_episode 600 --test_query_num_per_class 3 \
    --gpu 1 --train_path /home/aistudio/work/mini_Imagenet/dataset/train \
    --val_path /home/aistudio/work/mini_Imagenet/dataset/val --model_path ./model/best_accuracy

使用测试集验证  one shot learing 准确率。</br>

    python -u test.py --c_way 5 --k_shot 1 \
    --episode 10 --test_episode 600 --test_query_num_per_class 3 \
    --gpu 1 --test_path /home/aistudio/work/mini_Imagenet/dataset/test

运行 five shot learing方法如下。</br>

    python -u train.py --c_way 5 --k_shot 5 --query_num_per_class 10 \
    --episode 100000 --test_episode 600 --test_query_num_per_class 5 \
    --gpu 1 --train_path /home/aistudio/work/mini_Imagenet/dataset/train \
    --val_path /home/aistudio/work/mini_Imagenet/dataset/val --model_path ./model_few_shot/best_accuracy

使用测试集验证 five shot learing 准确率。</br>

    python -u test.py --c_way 5 --k_shot 5 \
    --episode 10 --test_episode 600 --test_query_num_per_class 5 \
    --gpu 1 --test_path /home/aistudio/work/mini_Imagenet/dataset/test \
    --model_path ./model_few_shot/best_accuracy
