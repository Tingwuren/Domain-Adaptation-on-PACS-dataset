# Homework 3: Deep Domain Adaptation

# 作业3:深度域适应

Mirco Planamente

米尔科·普拉纳门特

mirco.planamente@polito.it

Originally made by

原作者

Antonio D’Innocente

安东尼奥·迪诺森特

## Overview

## 概述

- The task is to implement DANN, a Domain Adaptation algorithm, on the PACS dataset using AlexNet

- 任务是使用AlexNet在PACS数据集上实现域适应算法DANN(深度对抗网络)

- Let's have a brief recap at Domain Adaptation and DANN

- 让我们简要回顾一下域适应和DANN

## The ideal case

## 理想情况

![bo_d2d0q677aajc738q0tjg_2_1344_554_627_536_0.jpg](images/bo_d2d0q677aajc738q0tjg_2_1344_554_627_536_0.jpg)

![bo_d2d0q677aajc738q0tjg_2_313_597_627_488_0.jpg](images/bo_d2d0q677aajc738q0tjg_2_313_597_627_488_0.jpg)

## The ideal case

## 理想情况

- Data has been collected and annotated in ideal conditions

- 数据是在理想条件下收集和标注的

- Test data is from the same distribution of training data

- 测试数据与训练数据来自相同分布

- Accuracy: HIGH

- 准确率:高

![bo_d2d0q677aajc738q0tjg_3_949_753_874_535_0.jpg](images/bo_d2d0q677aajc738q0tjg_3_949_753_874_535_0.jpg)

## The not ideal case

## 非理想情况

![bo_d2d0q677aajc738q0tjg_4_1401_543_468_543_0.jpg](images/bo_d2d0q677aajc738q0tjg_4_1401_543_468_543_0.jpg)

## The not ideal case

## 非理想情况

- Annotated data and unlabeled data don't match

- 标注数据和未标注数据不匹配

- Test data is from a different distribution of training data

- 测试数据与训练数据来自不同分布

- Accuracy: LOW

- 准确性:低

![bo_d2d0q677aajc738q0tjg_5_1026_758_521_530_0.jpg](images/bo_d2d0q677aajc738q0tjg_5_1026_758_521_530_0.jpg)

## Domain Adaptation

## 领域适应

- Align training features with test features

- 使训练特征与测试特征对齐

![bo_d2d0q677aajc738q0tjg_6_393_706_1596_543_0.jpg](images/bo_d2d0q677aajc738q0tjg_6_393_706_1596_543_0.jpg)

## DANN

## 对抗性判别域适应(DANN)

- Align training features with test features

- 使训练特征与测试特征对齐

- Train a binary classifier to discriminate between source and target

- 训练一个二元分类器以区分源域和目标域

- Train a feature extractor to confuse the features such that it is hard for the binary classifier to distinguish source from target

- 训练一个特征提取器来混淆特征，以使二元分类器难以区分源域和目标域

## DANN

## 对抗性判别域适应(DANN)

![bo_d2d0q677aajc738q0tjg_8_338_544_1655_720_0.jpg](images/bo_d2d0q677aajc738q0tjg_8_338_544_1655_720_0.jpg)

## DANN: architecture

## 对抗性判别域适应(DANN):架构

- Feature extractor Gf

- 特征提取器Gf

- Fully Convolutional

- 全卷积

- Label predictor Gy

- 标签预测器Gy

- Trained on source labels

- 使用源标签进行训练

- Domain classifier Gd

- 域分类器Gd

- Must discriminate if an image comes from

- 必须判别图像来自

the source domain or the target domain

源域还是目标域

- gradient reversal layer

- 梯度反转层

- Inverts the gradients of $\mathrm{{Gd}}$

- 反转 $\mathrm{{Gd}}$ 的梯度

![bo_d2d0q677aajc738q0tjg_9_1154_599_1131_490_0.jpg](images/bo_d2d0q677aajc738q0tjg_9_1154_599_1131_490_0.jpg)

## DANN: architecture

## DANN:架构

- Gd is trained to discriminate between source and target

- 训练Gd以区分源域和目标域

- The gradient reversal layer inverts the gradient of Gd

- 梯度反转层反转Gd的梯度

- Gf now wants to maximize error of Gd

- Gf现在想要最大化Gd的误差

- The feature extractor Gf tries to fool the domain classifier Gd

- 特征提取器Gf试图欺骗域分类器Gd

- By generating domain invariant features

- 通过生成域不变特征

- Gy is trained on these features

- Gy在这些特征上进行训练

![bo_d2d0q677aajc738q0tjg_10_1153_602_1132_487_0.jpg](images/bo_d2d0q677aajc738q0tjg_10_1153_602_1132_487_0.jpg)

## Homework 3

## 作业3

0 - Before starting

0 - 开始之前

1 - The Dataset

1 - 数据集

2 - Implementing the Model

2 - 实现模型

3 - Domain Adaptation

3 - 域适应

4 - (Extra) Cross Domain Validation

4 - (额外)跨域验证

## 0 - Before Starting

## 0 - 开始之前

- As with Homework 2, the assignment is in Colab

- 和作业2一样，本次作业在Colab上完成

- You are not provided a new starting template

- 不会提供新的起始模板

- However, you can start from the Homework 2 template

- 不过，你可以从作业2的模板开始

- https://colab.research.google.com/drive/1PhNPpklp9FbxJEtsZ8Jp9qXQa4aZDK5Y

- https://colab.research.google.com/drive/1PhNPpklp9FbxJEtsZ8Jp9qXQa4aZDK5Y

## 1 - The dataset

## 1 - 数据集

PACS is an image classification Dataset

PACS是一个图像分类数据集

![bo_d2d0q677aajc738q0tjg_13_1213_540_1063_599_0.jpg](images/bo_d2d0q677aajc738q0tjg_13_1213_540_1063_599_0.jpg)

- 7 classes

- 7个类别

- 4 domains

- 4个领域

- Photo, Art painting, Cartoon, Sketch

- 照片、艺术绘画、卡通、素描

The Dataset is provided here (you have to integrate it in the template):

数据集在此处提供(你必须将其集成到模板中):

https://github.com/MachineLearning2020/Home work3-PACS

https://github.com/MachineLearning2020/Home work3-PACS

## 1 - The dataset

## 1 - 数据集

Navigate

浏览

![bo_d2d0q677aajc738q0tjg_14_1214_540_1061_599_0.jpg](images/bo_d2d0q677aajc738q0tjg_14_1214_540_1061_599_0.jpg)

https://github.com/MachineLearning2020/Home work3-PACS/tree/master/PACS and explore the dataset structure

https://github.com/MachineLearning2020/Home work3-PACS/tree/master/PACS并探索数据集结构

As you will see, images are organized in folders

如你所见，图像按文件夹组织

Hint: You can easily read each domain as a PyTorch dataset using the ImageFolder class

提示:使用ImageFolder类，你可以轻松地将每个领域读取为PyTorch数据集

## 2 - Implementing the Model

## 2 - 模型实现

The original implementation of DANN is on small networks for digits datasets, we will try to implement it using AlexNet

DANN的原始实现是针对数字数据集的小型网络，我们将尝试使用AlexNet来实现它

![bo_d2d0q677aajc738q0tjg_15_1154_588_1103_473_0.jpg](images/bo_d2d0q677aajc738q0tjg_15_1154_588_1103_473_0.jpg)

Hint: Original implementations of DANN already exists in public PyTorch repositories

提示:DANN的原始实现已存在于公共的PyTorch代码库中

## 2 - Implementing the Model

## 2 - 模型实现

You have to implement DANN in PyTorch in this way:

你必须以这种方式在PyTorch中实现DANN:

- Gf as the AlexNet convolutional layers

- 将Gf作为AlexNet卷积层

- Gy as the AlexNet fully connected layers

- 将Gy作为AlexNet全连接层

- (up to this point it is a standard AlexNet)

- (到此为止它是一个标准的AlexNet)

- Gd as a separate branch with the same architecture of AlexNet fully connected layers (classifier)

- 将Gd作为具有与AlexNet全连接层(分类器)相同架构的单独分支

- Basically, you have to add a new densely connected branch with 2 output neurons

- 基本上，你必须添加一个具有2个输出神经元的新的密集连接分支

![bo_d2d0q677aajc738q0tjg_16_1154_588_1103_471_0.jpg](images/bo_d2d0q677aajc738q0tjg_16_1154_588_1103_471_0.jpg)

## 2 - Implementing the Model

## 2 - 模型实现

To implement Gd, you have to modify the init

要实现Gd，你必须修改AlexNet的初始化

function of AlexNet, and add the new branch

函数，并添加新分支

A. Modify the AlexNet's init function and add a

A. 修改AlexNet的初始化函数并添加一个

new classifier, Gd identical to the AlexNet

新的分类器，Gd与AlexNet

classifier

分类器相同

B. Both the original network and the new

B. 原始网络和新的

classifier must be initialized with the

分类器都必须用

weights of ImageNet (Gy and Gd will have

ImageNet的权重初始化(Gy和Gd将具有

the same starting weights)

相同的初始权重)

![bo_d2d0q677aajc738q0tjg_17_1156_585_1101_476_0.jpg](images/bo_d2d0q677aajc738q0tjg_17_1156_585_1101_476_0.jpg)

## 2 - Implementing the Model

## 2 - 实现模型

Hint: Start from the AlexNet source code in https://github.com/pytorch/vision/blob/master/torch vision/models/alexnet.py (you might need to adjust some imports)

提示:从https://github.com/pytorch/vision/blob/master/torch vision/models/alexnet.py中的AlexNet源代码开始(可能需要调整一些导入)

![bo_d2d0q677aajc738q0tjg_18_1154_583_1103_478_0.jpg](images/bo_d2d0q677aajc738q0tjg_18_1154_583_1103_478_0.jpg)

If you create a new branch in the init function, and try to preload weights into the original branches, it gives you an error

如果你在init函数中创建一个新分支，并尝试将权重预加载到原始分支中，会出现错误

- Use the flag strict=False in the load_state_dict function to avoid this error

- 在load_state_dict函数中使用标志strict=False来避免此错误

## 2 - Implementing the Model

## 2 - 实现模型

Hint: Start from the AlexNet source code in https://github.com/pytorch/vision/blob/master/torch vision/models/alexnet.py (you might need to adjust some imports)

提示:从https://github.com/pytorch/vision/blob/master/torch vision/models/alexnet.py中的AlexNet源代码开始(可能需要调整一些导入)

![bo_d2d0q677aajc738q0tjg_19_1155_583_1103_478_0.jpg](images/bo_d2d0q677aajc738q0tjg_19_1155_583_1103_478_0.jpg)

- After you preload ImageNet weights in the original branches, copy weights of the original classifier into the new classifier

- 在原始分支中预加载ImageNet权重后，将原始分类器的权重复制到新的分类器中

Hint: You can access fc6 weights and biases with model.classifier[1].weight.data and model.classifier[1].bias.data

提示:你可以使用model.classifier[1].weight.data和model.classifier[1].bias.data访问fc6的权重和偏差

## 2 - Implementing the Model

## 2 - 实现模型

To implement the DANN logic you will have to change the forward function

要实现DANN逻辑，你必须更改forward函数

![bo_d2d0q677aajc738q0tjg_20_1155_586_1103_473_0.jpg](images/bo_d2d0q677aajc738q0tjg_20_1155_586_1103_473_0.jpg)

C. Implement a flag in the forward function (a boolean or something else) that you pass along with data to indicate if a batch of data must go to Gd or Gy

C. 在forward函数中实现一个标志(布尔值或其他类型)，你将其与数据一起传递，以指示一批数据是必须进入Gd还是Gy

a. If data goes to Gd, the gradient has to be reverted with the Gradient Reversal layer

a. 如果数据进入Gd，则必须使用梯度反转层反转梯度

Hint: you find an example of gradient reversal in: https://github.com/MachineLearning2020/Homework3- PACS/blob/master/gradient reversal example.py

提示:你可以在以下链接找到梯度反转的示例:https://github.com/MachineLearning2020/Homework3- PACS/blob/master/gradient reversal example.py

## 2 - Implementing the Model

## 2 - 模型实现

Hint: you find an example of gradient reversal in: https://github.com/MachineLearning2020/Homework3- PACS/blob/master/gradient reversal example.py

提示:你可以在以下链接找到梯度反转的示例:https://github.com/MachineLearning2020/Homework3- PACS/blob/master/gradient reversal example.py

![bo_d2d0q677aajc738q0tjg_21_1155_583_1103_478_0.jpg](images/bo_d2d0q677aajc738q0tjg_21_1155_583_1103_478_0.jpg)

As you may notice from the example, the reversal is multiplied by an alpha factor in the forward

从示例中你可能会注意到，在正向传播中，反转操作会乘以一个α因子

This is the weight of the reversed backpropagation, and must be optimized as an hyperparameter of the algorithm

这是反向传播反转的权重，必须作为算法的超参数进行优化

## 3 - Domain Adaptation

## 3 - 域适应

For this Homework, the source domain is Photo, while the target domain is Art painting

对于本次作业，源域是照片，目标域是艺术绘画

![bo_d2d0q677aajc738q0tjg_22_1215_540_1058_599_0.jpg](images/bo_d2d0q677aajc738q0tjg_22_1215_540_1058_599_0.jpg)

A. Train on Photo, and Test on Art painting without adaptation

A. 在照片上训练，在未进行适应的情况下在艺术绘画上测试

B. Train DANN on Photo and test on Art painting with DANN adaptation

B. 在照片上训练DANN，并在使用DANN适应的情况下在艺术绘画上测试

C. Compare results

C. 比较结果

## 3B - Implementing training with DANN

## 3B - 使用DANN实现训练

The network must be trained jointly on the labeled task (Photo) and the unsupervised task (discriminating between Photo and Art painting), and then tested on Art painting

网络必须在有标签任务(照片)和无监督任务(区分照片和艺术绘画)上联合训练，然后在艺术绘画上进行测试

![bo_d2d0q677aajc738q0tjg_23_1156_585_1103_476_0.jpg](images/bo_d2d0q677aajc738q0tjg_23_1156_585_1103_476_0.jpg)

- Divide a single training iteration in three steps that you execute sequentially before calling optimizer.step(   )

- 在调用optimizer.step(   )之前，将单个训练迭代分为三个步骤并按顺序执行

## 3B - Implementing training with DANN

## 3B - 使用DANN实现训练

1. train on source labels by forwarding source data to Gy, get the loss, and update gradients with loss.backward(   )

1. 通过将源数据转发到Gy来在源标签上进行训练，获取损失，并使用loss.backward(   )更新梯度

![bo_d2d0q677aajc738q0tjg_24_1156_588_1103_473_0.jpg](images/bo_d2d0q677aajc738q0tjg_24_1156_588_1103_473_0.jpg)

2. train the discriminator by forwarding source data to Gd, get the loss (the label is 0 for all data), and update gradients with loss.backward(   )

2. 通过将源数据转发到Gd来训练判别器，获取损失(所有数据的标签为0)，并使用loss.backward(   )更新梯度

3. train the discriminator by forwarding target data to Gd, get the loss (the label is 1), and update gradients with loss.backward(   )

3. 通过将目标数据输入Gd来训练判别器，得到损失值(标签为1)，并使用loss.backward(   )更新梯度

## 3B - Implementing training with DANN

## 3B - 使用DANN进行训练

Hint: 2. and 3. are binary classification steps (the discriminator must tell if the images comes from source or target), so you can use the nn.CrossEntropyLoss(   ) function as your criterion

提示:2. 和3. 是二分类步骤(判别器必须判断图像是来自源域还是目标域)，所以你可以使用nn.CrossEntropyLoss(   )函数作为准则

![bo_d2d0q677aajc738q0tjg_25_1155_579_1104_482_0.jpg](images/bo_d2d0q677aajc738q0tjg_25_1155_579_1104_482_0.jpg)

## 3B - Implementing training with DANN

## 3B - 使用DANN进行训练

After doing the 3 steps you call optimizer.step(   ) to apply gradients

完成这三个步骤后，调用optimizer.step(   )来应用梯度

![bo_d2d0q677aajc738q0tjg_26_1157_588_1102_473_0.jpg](images/bo_d2d0q677aajc738q0tjg_26_1157_588_1102_473_0.jpg)

Hint: For the second step you can use the same data you used for Gy

提示:对于第二步，你可以使用与Gy相同的数据

Hint: For the third step, you have to use a separate dataloader that iterates over the target dataset https://github.com/pytorch/pytorch/issues/1917

提示:对于第三步，你必须使用一个单独的数据加载器，它遍历目标数据集https://github.com/pytorch/pytorch/issues/1917

## 3 - Additional informations

## 3 - 附加信息

- For this task, use the entire Photo dataset for training, and the entire Art painting dataset for both training (without labels) and testing

- 对于此任务，使用整个照片数据集进行训练，并使用整个艺术绘画数据集进行训练(无标签)和测试

- This means that the adaptation set and the test set coincide (transductive)

- 这意味着适配集和测试集是重合的(转导)

- we don't do validation (more on this later)

- 我们不进行验证(稍后会详细说明)

![bo_d2d0q677aajc738q0tjg_27_1214_540_1059_599_0.jpg](images/bo_d2d0q677aajc738q0tjg_27_1214_540_1059_599_0.jpg)

## 3 - Additional informations

## 3 - 附加信息

If you are not going to do the extra point 4, run hyperparameter optimization for both 3A and 3B to optimize performances with some hyperparameter search algorithm

如果你不打算做额外的第4点，对3A和3B都运行超参数优化，使用一些超参数搜索算法来优化性能

![bo_d2d0q677aajc738q0tjg_28_1214_540_1061_599_0.jpg](images/bo_d2d0q677aajc738q0tjg_28_1214_540_1061_599_0.jpg)

- Besides usual hyperparameters, you have to optimize alpha

- 除了常见的超参数，你还必须优化alpha

- (This is essentially cheating)

- (这本质上是作弊)

- You can use the same batch size for the Photo dataloader and the Art painting dataloader

- 你可以对照片数据加载器和艺术绘画数据加载器使用相同的批量大小

## 4 - (Extra) - Cross Domain Validation

## 4 - (附加) - 跨域验证

As you noticed, we didn't do the validation step

如你所见，我们没有进行验证步骤

![bo_d2d0q677aajc738q0tjg_29_1212_540_1061_599_0.jpg](images/bo_d2d0q677aajc738q0tjg_29_1212_540_1061_599_0.jpg)

- Validation is an open problem in Domain Adaptation

- 验证是域适应中的一个开放性问题

- However, without validation, we are looking at results in the test set (cheating)

- 然而，没有验证的话，我们看到的测试集结果是有问题的(作弊行为)

- If you want to complete the extra assignment: do 3A and 3B by validating hyperparameters

- 如果要完成附加作业:通过验证超参数来完成3A和3B

- How can you validate Photo to Art painting transfer if we don't have Art painting labels?

- 如果我们没有艺术绘画标签，如何验证照片到艺术绘画的转换？

## 4 - (Extra) - Cross Domain Validation

## 4 - (附加) - 跨域验证

- We validate hyperparameters by measuring performances on Photo to Cartoon transfer and Photo to Sketch transfer

- 我们通过测量照片到卡通转换和照片到素描转换的性能来验证超参数

![bo_d2d0q677aajc738q0tjg_30_1212_540_1061_599_0.jpg](images/bo_d2d0q677aajc738q0tjg_30_1212_540_1061_599_0.jpg)

A. Run a grid search (or other hyperparameter search method) on Photo to Cartoon and Photo to Sketch, without Domain Adaptation, and average results for each set of hyperparameters

A. 在不进行域适应的情况下，对照片到卡通和照片到素描进行网格搜索(或其他超参数搜索方法)，并对每组超参数的结果求平均值

B. Implement 3A with the best

B. 使用在4A中找到的最佳超参数实现3A

hyperparameters found in 4A

超参数

## 4 - (Extra) - Cross Domain Validation

## 4 - (附加) - 跨域验证

- We validate hyperparameters by measuring performances on Photo to Cartoon transfer and Photo to Sketch transfer

- 我们通过测量照片到卡通转换和照片到素描转换的性能来验证超参数

![bo_d2d0q677aajc738q0tjg_31_1212_540_1061_599_0.jpg](images/bo_d2d0q677aajc738q0tjg_31_1212_540_1061_599_0.jpg)

C. Run a grid search (or other hyperparameter search method) on Photo to Cartoon and

C. 在进行域适应的情况下，对照片到卡通和照片到素描进行网格搜索(或其他超参数搜索方法)，并对每组超参数的结果求平均值

Photo to Sketch, with Domain Adaptation, and average results for each set of hyperparameters

照片到素描，并对每组超参数的结果求平均值

D. Implement 3B with the best

D. 使用最佳的实现3B

hyperparameters found in 4C

在4C中找到的超参数

## 4 - (Extra) - Cross Domain Validation

## 4 - (额外) - 跨域验证

If you do the validation step, you don’t have to optimize hyperparameters of 3A and 3B on the test set

如果执行验证步骤，则无需在测试集上优化3A和3B的超参数

![bo_d2d0q677aajc738q0tjg_32_1212_540_1064_599_0.jpg](images/bo_d2d0q677aajc738q0tjg_32_1212_540_1064_599_0.jpg)

## Submission rules

## 提交规则

- Deadline: Two week before the first round (NOT the first you enrol in).

- 截止日期:第一轮前两周(不是你报名的第一轮)。

- Uploading: through "Portale della didattica"

- 上传:通过“Portale della didattica”

Submit a zip named <YOUR_ID>_homework3.zip. The zip should contain two items:

提交一个名为<YOUR_ID>_homework3.zip的压缩包。该压缩包应包含两项内容:

- A pdf report describing data, your implementation choices, results and discussions

- 一份pdf报告，描述数据、你的实现选择、结果和讨论

- Code

- 代码

## FAQ

## 常见问题解答

1. Should I implement hyperparameter search in the code or it is fine to test manually?

1. 我应该在代码中实现超参数搜索还是手动测试就可以？

- Due to Colab limitations, it is ok to test hyperparameters by manually changing them in the code

- 由于Colab的限制，可以通过在代码中手动更改超参数来测试它们

2. The Discriminator loss doesn't increase

2. 判别器损失没有增加

- The feature extractor tries to maximize the loss of the discriminator

- 特征提取器试图最大化判别器的损失

- However, the Discriminator is still trying to minimize its own loss

- 然而，判别器仍在试图最小化其自身的损失

3. Results with DANN have small or no improvements

3. 使用DANN的结果改进很小或没有改进

- Even 1-2% increases on PACS are significative

- 即使PACS(Picture Archiving and Communication System，图像存档与通信系统)增加1 - 2%也是有意义的

- The implementation might not yield adaptation results in this setting

- 在这种情况下，该实施可能不会产生适应性结果

- Optional things that you can try:

- 你可以尝试的可选事项:

i. Advanced exponential scheduling for alpha (check DANN paper)

i. 针对α的高级指数调度(查看DANN论文)

ii. Different implementations for the discriminator (experiment, check DANN paper)

ii. 鉴别器的不同实现方式(进行实验，查看DANN论文)

iii. Implement the exact AlexNet architecture in pytorch, load pre-trained weights from the original caffe

iii. 在PyTorch中实现精确的AlexNet架构，从原始Caffe模型加载预训练权重，调整输入以适应PyTorch预处理([0 - 255] -> [0 - 1]，BGR -> RGB)

model, adjust inputs to fit pytorch preprocessing ([0-255] -> [0-1], BGR -> RGB)

模型，调整输入以适应PyTorch预处理([0 - 255] -> [0 - 1]，BGR -> RGB)