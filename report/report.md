Machine Learning and Deep Learning

机器学习与深度学习

Report - Homework 3

报告 - 作业3

Roberto Franceschi (s276243)

罗伯托·弗朗切斯基(s276243)

## 1 Introduction

## 1 引言

In this report, we will explore an image classification problem using different domains. In all applications of CNNs experimented during this course, we assumed that our training data representative of the underlying distribution. However, if the inputs at test time differ significantly from the training data, the model might not perform very well. For this homework, we have no access at training time to the distribution (i.e. domain) where the test images are taken from. In order to tackle the issue, we will use a modified version of the AlexNet [1] deep neural network structure that allows not only to classify input images in the source domain but also to transfer this capability to the target domain. Such task is called domain adaptation (DA) in deep learning research and has several applications in real-life situations.

在本报告中，我们将使用不同领域的数据来探索一个图像分类问题。在本课程中进行的所有卷积神经网络(CNN)实验应用中，我们假设训练数据代表了潜在分布。然而，如果测试时的输入与训练数据有显著差异，模型可能表现不佳。对于本次作业，我们在训练时无法获取测试图像所来自的分布(即领域)。为了解决这个问题，我们将使用AlexNet [1] 深度神经网络结构的一个修改版本，它不仅能够对源领域中的输入图像进行分类，还能将这种能力迁移到目标领域。这种任务在深度学习研究中被称为领域自适应(DA)，并且在现实生活中有多种应用。

For this analysis we will use the PACS dataset [2], which contains overall 9991 images, split unevenly between 7 classes and 4 domains: Art painting, Cartoon, Photo, Sketch. Figure 1 shows as an example some images of class 'horse' taken from this dataset with the correspondent domain label. As can be clearly seen from the example there are lots of differences between domains.

对于此分析，我们将使用PACS数据集 [2]，该数据集总共包含9991张图像，在7个类别和4个领域之间不均匀分布:艺术绘画、卡通、照片、素描。图1以示例的形式展示了从该数据集中选取的一些“马”类别的图像以及相应的领域标签。从示例中可以清楚地看到不同领域之间存在很多差异。

Our goal will be to train the network on images from the Photo domain and then to be able to correctly classify samples from the Art domain. Furthermore, Cross Domain Validation will be taken into account in the attempt of finding the best hyperparameters to increase the overall performance of the net.

我们的目标是在照片领域的图像上训练网络，然后能够正确分类来自艺术领域的样本。此外，在寻找最佳超参数以提高网络整体性能的尝试中，将考虑跨领域验证。

![bo_d2d0r6v7aajc738q0tqg_0_197_1604_1255_387_0.jpg](images/bo_d2d0r6v7aajc738q0tqg_0_197_1604_1255_387_0.jpg)

Figure 1: Sample images from the PACS dataset one for each domain: (a) photo, (b) art painting, (c) cartoon, (d) sketch

图1:PACS数据集中每个领域的示例图像:(a)照片，(b)艺术绘画，(c)卡通，(d)素描

Additionally, in Table 1 and Figure 2 are summarized the distributions of images per domain and classes. The original size of each image is ${227} \times  {227}$ pixels (with 3 channels, i.e. RGB) and their distribution across the 7 classes has the following parameters: mean of about 1427 images per class and a standard deviation of 269.4 images per class.

此外，表1和图2总结了每个领域和类别的图像分布。每张图像的原始大小为 ${227} \times  {227}$ 像素(具有3个通道，即RGB)，它们在7个类别中的分布具有以下参数:每个类别平均约1427张图像，标准差为每个类别269.4张图像。

It is also interesting to note that the Photo dataset, the one we will use for training the neural network, is the smallest one, only 1670 samples (wrt a mean of 2497,7 images per domain).

同样值得注意的是，我们将用于训练神经网络的照片数据集是最小的，只有1670个样本(相对于每个领域平均2497.7张图像)。

<table><tr><td/><td>Dog</td><td>Elephant</td><td>Giraffe</td><td>Guitar</td><td>Horse</td><td>House</td><td>Person</td><td>Total domain</td></tr><tr><td>Photo</td><td>189</td><td>202</td><td>182</td><td>186</td><td>199</td><td>280</td><td>432</td><td>1670</td></tr><tr><td>Art Painting</td><td>379</td><td>255</td><td>285</td><td>184</td><td>201</td><td>295</td><td>449</td><td>2048</td></tr><tr><td>Cartoon</td><td>389</td><td>255</td><td>285</td><td>184</td><td>201</td><td>295</td><td>449</td><td>2344</td></tr><tr><td>Sketch</td><td>772</td><td>740</td><td>753</td><td>608</td><td>816</td><td>80</td><td>160</td><td>3929</td></tr><tr><td>Total class</td><td>1729</td><td>1654</td><td>1566</td><td>1113</td><td>1540</td><td>943</td><td>1446</td><td/></tr></table>

<table><tbody><tr><td></td><td>狗</td><td>大象</td><td>长颈鹿</td><td>吉他</td><td>马</td><td>房子</td><td>人</td><td>总领域</td></tr><tr><td>照片</td><td>189</td><td>202</td><td>182</td><td>186</td><td>199</td><td>280</td><td>432</td><td>1670</td></tr><tr><td>艺术绘画</td><td>379</td><td>255</td><td>285</td><td>184</td><td>201</td><td>295</td><td>449</td><td>2048</td></tr><tr><td>卡通</td><td>389</td><td>255</td><td>285</td><td>184</td><td>201</td><td>295</td><td>449</td><td>2344</td></tr><tr><td>素描</td><td>772</td><td>740</td><td>753</td><td>608</td><td>816</td><td>80</td><td>160</td><td>3929</td></tr><tr><td>总类别</td><td>1729</td><td>1654</td><td>1566</td><td>1113</td><td>1540</td><td>943</td><td>1446</td><td></td></tr></tbody></table>

Table 1: Number of images in each class and each domain

表1:每个类别和每个领域中的图像数量

![bo_d2d0r6v7aajc738q0tqg_1_208_1230_1216_835_0.jpg](images/bo_d2d0r6v7aajc738q0tqg_1_208_1230_1216_835_0.jpg)

Figure 2: Distribution of number of images in each class and each domain

图2:每个类别和每个领域中图像数量的分布

## 2 Network setup

## 2网络设置

The basic idea behind the DANN implementation presented in [3] is to learn a model that can generalize well from one domain to another, while ensuring that the internal representation of the neural network contains no discriminative information about the origin of the input (source or target). As already mentioned above, to implement this idea we will use a modified version of the original AlexNet. In particular, we have two parallel branches of fully connected layers after the convolutional section. The first one is the one from the original net with an output of 7-dimensional vector, corresponding to the classes in the PACS dataset.

[3]中提出的DANN实现背后的基本思想是学习一个能够很好地从一个领域推广到另一个领域的模型，同时确保神经网络的内部表示不包含关于输入来源(源或目标)的判别信息。如前所述，为了实现这个想法，我们将使用原始AlexNet的一个修改版本。具体来说，在卷积部分之后，我们有两个并行的全连接层分支。第一个是来自原始网络的分支，其输出是一个7维向量，对应于PACS数据集中的类别。

Instead, the parallel branch aims to determine whether an image comes from the source or the target domain and it is connected to the feature extractor via a gradient reversal layer that multiplies the gradient by a certain negative constant during the backpropagation-based training. So, it will have as output a 2d vector, since it plays the role of a binary classifier.

相反，并行分支旨在确定图像来自源域还是目标域，并且它通过一个梯度反转层连接到特征提取器，该梯度反转层在基于反向传播的训练期间将梯度乘以某个负常数。因此，它将输出一个二维向量，因为它起到二元分类器的作用。

However, during training we will try to confuse the network so that it is not able to tell the difference between the domains: by doing so during classification we can indifferently provide images from any domain to the network and expect that they are classified correctly. The following figure shows what is described above, in particular includes a feature extractor (blue) of the original AlexNet, a label classifier (blue) and a domain classifier (pink).

然而，在训练期间，我们将尝试混淆网络，使其无法区分不同的域:通过在分类期间这样做，我们可以无差别地向网络提供来自任何域的图像，并期望它们被正确分类。下图展示了上述内容，特别包括原始AlexNet的一个特征提取器(蓝色)、一个标签分类器(蓝色)和一个域分类器(粉色)。

![bo_d2d0r6v7aajc738q0tqg_2_208_1041_1244_531_0.jpg](images/bo_d2d0r6v7aajc738q0tqg_2_208_1041_1244_531_0.jpg)

Figure 3: Domain-adversarial neural network architecture by Ganin et al. [3]

图3:Ganin等人[3]提出的域对抗神经网络架构

Before feeding the network, the desired transformation has been computed on the input images, normalizing them, and adapting their size to the input required by AlexNet, which is ${224} \times  {224}$ pixels (using a transforms. CenterCrop).

在将图像输入网络之前，已经对输入图像进行了所需的变换，对它们进行归一化，并将其大小调整为AlexNet所需的输入大小，即 ${224} \times  {224}$ 像素(使用transforms.CenterCrop)。

The PACS dataset and code, together with results, can be found at: https://github.com/robertofranceschi/DANN.

PACS数据集、代码以及结果可在以下网址找到:https://github.com/robertofranceschi/DANN。

## 3 Implementation of DANN adaptation

## 3 DANN适应的实现

To implement what is described in Section 2 we define a new class DANN_AlexNet which implements the network, just like the class AlexNet in PyTorch. This class define the blocks of the network: the first will contain the convolutional layers (self.features), while the other two will have the two parallel fully-connected layer (self.classifier for the label classifier and self. GD for the discriminator). Afterwards we have to change the logic of the forward function in order to allow to take the parallel branch when needed (this is done introducing the parameter alpha). When alpha is None we assume that we are training with supervision, instead if we pass alpha, we are training the discriminator. Likewise, the backward function has to propagate the gradient from the parallel branch back to the convolutional layers.

为了实现第2节中描述的内容，我们定义了一个新类DANN_AlexNet，它实现了该网络，就像PyTorch中的AlexNet类一样。这个类定义了网络的各个模块:第一个模块将包含卷积层(self.features)，而另外两个模块将有两个并行的全连接层(self.classifier用于标签分类器，self.GD用于判别器)。之后，我们必须更改前向函数的逻辑，以便在需要时能够使用并行分支(这是通过引入参数alpha来完成的)。当alpha为None时，我们假设我们正在进行有监督的训练，相反，如果我们传入alpha，我们就是在训练判别器。同样，反向函数必须将来自并行分支的梯度传播回卷积层。

Furthermore, both the original network classifier and the new domain classifier has been initialized with the weights of the network as in the version of the AlexNet pretrained on the ImageNet dataset. In order to do this it is sufficient to load the parameters by means of the function load state dict provided in the original implementation of pytorch.

此外，原始网络分类器和新的域分类器都已使用在ImageNet数据集上预训练的AlexNet版本中的网络权重进行初始化。为此，通过pytorch原始实现中提供的load_state_dict函数加载参数就足够了。

## 4 Domain adaptation using art painting as target

## 4以艺术绘画为目标的域适应

At the beginning, we train a traditional AlexNet and see how it performs in case of a domain adaptation problem. In this first part, runs were carried out without considering the cartoon and sketch domains as validation sets. Therefore, the network was trained on the photo domain and tested on art painting without domain adaptation (point $3\mathrm{\;A}$ in the assignment) and consequently with DANN adaptation (point 3B). The results regarding this part are shown in the following table. Clearly, we see from the last column, as we expected, that we have a significant increment performing training with DANN adaptation.

一开始，我们训练一个传统的AlexNet，并观察它在域适应问题中的表现。在这第一部分中，运行时没有将卡通和草图域视为验证集。因此，网络在照片域上进行训练，并在没有域适应的情况下在艺术绘画上进行测试(作业中的点 $3\mathrm{\;A}$)，并因此在有DANN适应的情况下进行测试(点3B)。关于这部分的结果如下表所示。显然，从最后一列中我们可以看到，正如我们所期望的，使用DANN适应进行训练有显著的提升。

<table><tr><td/><td>Test accuracy without DANN</td><td>Test accuracy with DANN</td><td>% Increment</td></tr><tr><td>${LR} = {5e} - 3$ ${BS} = {256}$</td><td>51.35 % (± 1.0%)</td><td>54.52 % (± 0.1%)</td><td>+3,17%</td></tr><tr><td>${LR} = {5e} - 3$ ${BS} = {128}$</td><td>51.90 % (± 1.7%)</td><td>52.46 % (± 2.8%)</td><td>+ 0,56%</td></tr><tr><td>${LR} = {1e} - 3$ ${BS} = {256}$</td><td>49.33 % (±0.5%)</td><td>51.10 % (± 0.8%)</td><td>+ 1.77%</td></tr><tr><td>${LR} = {1e} - 3$ ${BS} = {128}$</td><td>50.03 % (± 0.3%)</td><td>51.60 % (± 1.8%)</td><td>+ 1.57%</td></tr><tr><td>${LR} = {1e} - 2$ ${BS} = {256}$</td><td>50.19 % (± 2.4%)</td><td>50.20 % (± 3.2%)</td><td>+0.01 %</td></tr><tr><td>${LR} = {1e} - 2$ ${BS} = {128}$</td><td>51.18 % (± 1.7%)</td><td>53.59 % (± 2.1%)</td><td>+ 2.40%</td></tr></table>

<table><tbody><tr><td></td><td>无对抗神经网络的测试准确率</td><td>有对抗神经网络的测试准确率</td><td>增长率</td></tr><tr><td>${LR} = {5e} - 3$ ${BS} = {256}$</td><td>51.35%(±1.0%)</td><td>54.52%(±0.1%)</td><td>+3,17%</td></tr><tr><td>${LR} = {5e} - 3$ ${BS} = {128}$</td><td>51.90%(±1.7%)</td><td>52.46%(±2.8%)</td><td>+ 0,56%</td></tr><tr><td>${LR} = {1e} - 3$ ${BS} = {256}$</td><td>49.33%(±0.5%)</td><td>51.10%(±0.8%)</td><td>+ 1.77%</td></tr><tr><td>${LR} = {1e} - 3$ ${BS} = {128}$</td><td>50.03%(±0.3%)</td><td>51.60%(±1.8%)</td><td>+ 1.57%</td></tr><tr><td>${LR} = {1e} - 2$ ${BS} = {256}$</td><td>50.19%(±2.4%)</td><td>50.20%(±3.2%)</td><td>+0.01 %</td></tr><tr><td>${LR} = {1e} - 2$ ${BS} = {128}$</td><td>51.18%(±1.7%)</td><td>53.59%(±2.1%)</td><td>+ 2.40%</td></tr></tbody></table>

Table 2: Comparison of different hyperparameter sets without adaptation and with DANN adaptation. The last column report the increment between the two presented strategies.

表2:不同超参数集在无自适应和有DANN自适应情况下的比较。最后一列报告了所呈现的两种策略之间的增量。

Note that in the second case we did something equivalent to "cheating" in the field of machine learning because informations regarding the test domain (i.e. art painting) were used during training of the neural net (in the discriminator branch). As we will see in the next section different strategies has been implemented that make use of informations taken from the other domains (cartoon and sketch).

请注意，在第二种情况下，我们在机器学习领域做了相当于“作弊”的事情，因为在神经网络训练期间(在判别器分支中)使用了有关测试域(即艺术绘画)的信息。正如我们将在下一节中看到的，已经实施了不同的策略，这些策略利用了来自其他域(卡通和草图)的信息。

## 5 Cross Domain Validation without adaptation

## 5 无自适应的跨域验证

In this second step of our analysis we will consider also the others domains as validation sets. So, we will train the network on images coming from the source domain (i.e. Photo) and then we will test it on the target domain (i.e. art painting), without giving any informations to the network about the difference between the domains (i.e., without DANN adaptation).

在我们分析的第二步中，我们也将把其他域视为验证集。因此，我们将在来自源域(即照片)的图像上训练网络，然后在目标域(即艺术绘画)上对其进行测试，而不向网络提供任何有关域之间差异的信息(即，无DANN自适应)。

The hyperparameters we are going to tune in this first part are learning rate and batch size. The other hyperparameters are fixed and have the following values:

我们将在第一部分中调整的超参数是学习率和批量大小。其他超参数是固定的，具有以下值:

- Epochs: 30

- 轮次:30

- Step size: 20

- 步长:20

- Weight decay: $5 \cdot  {10}^{-5}$

- 权重衰减:$5 \cdot  {10}^{-5}$

- Gamma: 0.1

- 伽马:0.1

They define respectively how many times we will pass through the entire dataset in the course of the training phase, the number of epochs after which the learning rate is decreased, the regularization parameter and the decreasing factor of the learning rate. The optimizer used in our analysis is the SGD with momentum.

它们分别定义了在训练阶段我们将遍历整个数据集的次数、学习率降低之前的轮次数量、正则化参数以及学习率的降低因子。我们分析中使用的优化器是带动量的随机梯度下降(SGD)。

For each of the two validation domains we perform a grid search across different values of the hyperparameters learning rate (LR) and batch size (BS). The results of the validation phase are shown in table 3 for the cartoon and sketch domain. In order to properly select the best hyperparameter set, we pick the results obtained on the two domains and choose the set that performed at best on average (see Table 4).

对于两个验证域中的每一个，我们针对超参数学习率(LR)和批量大小(BS)的不同值进行网格搜索。验证阶段的结果在表3中显示了卡通和草图域的情况。为了正确选择最佳超参数集，我们选取在两个域上获得的结果，并选择平均表现最佳的集合(见表4)。

<table><tr><td/><td>${BS} = {256}$</td><td>${BS} = {128}$</td></tr><tr><td>${LR} = {5e} - 3$</td><td>27.56 %</td><td>26.78 %</td></tr><tr><td>${LR} = {1e} - 3$</td><td>24.59 %</td><td>26.02 %</td></tr><tr><td>${LR} = {1e} - 2$</td><td>28.65 % (a)</td><td>27.54 %</td></tr></table>

<table><tr><td/><td>${BS} = {256}$</td><td>${BS} = {128}$</td></tr><tr><td>${LR} = {5e} - 3$</td><td>27.56 %</td><td>26.78 %</td></tr><tr><td>${LR} = {1e} - 3$</td><td>24.59 %</td><td>26.02 %</td></tr><tr><td>${LR} = {1e} - 2$</td><td>28.65 % (a)</td><td>27.54 %</td></tr></table>

<table><tr><td/><td>${BS} = {256}$</td><td>${BS} = {128}$</td></tr><tr><td>${LR} = {5e} - 3$</td><td>29.34 %</td><td>30.86 %</td></tr><tr><td>${LR} = {1e} - 3$</td><td>30.27 %</td><td>29.91 %</td></tr><tr><td>${LR} = {1e} - 2$</td><td>27.54 % (b)</td><td>33.26 %</td></tr></table>

<table><tr><td/><td>${BS} = {256}$</td><td>${BS} = {128}$</td></tr><tr><td>${LR} = {5e} - 3$</td><td>29.34 %</td><td>30.86 %</td></tr><tr><td>${LR} = {1e} - 3$</td><td>30.27 %</td><td>29.91 %</td></tr><tr><td>${LR} = {1e} - 2$</td><td>27.54 % (b)</td><td>33.26 %</td></tr></table>

Table 3: Comparison of different hyperparameter sets on the (a) cartoon and (b) sketch domains.

表3:不同超参数集在(a)卡通和(b)草图领域的比较。

<table><tr><td/><td>${BS} = {256}$</td><td>${BS} = {128}$</td></tr><tr><td>${LR} = {5e} - 3$</td><td>28.45 %</td><td>28.82 %</td></tr><tr><td>${LR} = {1e} - 3$</td><td>27.43 %</td><td>26.42 %</td></tr><tr><td>${LR} = {1e} - 2$</td><td>29.28 %</td><td>30.40 %</td></tr></table>

<table><tr><td/><td>${BS} = {256}$</td><td>${BS} = {128}$</td></tr><tr><td>${LR} = {5e} - 3$</td><td>28.45 %</td><td>28.82 %</td></tr><tr><td>${LR} = {1e} - 3$</td><td>27.43 %</td><td>26.42 %</td></tr><tr><td>${LR} = {1e} - 2$</td><td>29.28 %</td><td>30.40 %</td></tr></table>

Table 4: Average of the validation results between the cartoon and sketch domain.

表4:卡通与草图领域之间验证结果的平均值。

The best score is obtained with $\mathrm{{LR}} = 1\mathrm{e} - 2$ and batch size $= {128}$ . Finally, we test the network on the Art painting domain, which delivers an accuracy of 51.06% (±1.51%).

使用 $\mathrm{{LR}} = 1\mathrm{e} - 2$ 和批量大小 $= {128}$ 可获得最佳分数。最后，我们在艺术绘画领域对网络进行测试，其准确率为51.06%(±1.51%)。

## 6 Cross Domain Validation with domain adaptation

## 6 使用域适应的跨域验证

After implementing the modified neural network for domain adaptation, we are able to execute the task presented in the beginning of the report. In the training phase, our task is to optimize the parameters both of the label classifier and of the domain classifier. In order to train the discriminator, we use the images from the cartoon and sketch domains as target. Each training epoch consists of three steps as described in paper [3]:

在实现用于域适应的改进神经网络后，我们能够执行报告开头提出的任务。在训练阶段，我们的任务是优化标签分类器和域分类器的参数。为了训练鉴别器，我们使用来自卡通和草图领域的图像作为目标。每个训练轮次包括论文[3]中所述的三个步骤:

1. training the classifier using images from the source domain,

1. 使用源域的图像训练分类器，

2. training the discriminator using images from the source domain,

2. 使用源域的图像训练鉴别器，

3. training the discriminator using images from the target domain.

3. 使用目标域的图像训练鉴别器。

For this analysis we selected as hyperparameters to tune the same as before (LR and BS) and the parameter alpha, which is a weighting parameter to control the gradient propagation from the parallel branch. Therefore, we perform a grid search between the parameters of Section 4 and different values of alpha, that are 0.01, 0.1, 0.25, 0.5 and 0.8. As before, we validate the network on the domains cartoon and sketch in order to find the best hyperparameter set. The average results of cartoon and sketch are reported in table 5 .

对于此分析，我们选择与之前相同的超参数(学习率和批量大小)以及参数alpha进行调整，alpha是一个加权参数，用于控制来自并行分支的梯度传播。因此，我们在第4节的参数和不同的alpha值(0.01、0.1、0.25、0.5和0.8)之间进行网格搜索。与之前一样，我们在卡通和草图领域对网络进行验证，以找到最佳的超参数集。卡通和草图的平均结果报告在表5中。

<table><tr><td rowspan="2"/><td colspan="2">Alpha $= {0.01}$</td><td colspan="2">${Alpha} = {0.1}$</td><td colspan="2">Alpha $= {0.25}$</td></tr><tr><td>${BS} = {256}$</td><td>${BS} = {128}$</td><td>${BS} = {256}$</td><td>${BS} = {128}$</td><td>${BS} = {256}$</td><td>${BS} = {128}$</td></tr><tr><td>${LR} = {1e} - 4$</td><td>21.54 %</td><td>23.13 %</td><td>22.43 %</td><td>24.32 %</td><td>26.98 %</td><td>30.28 %</td></tr><tr><td>${LR} = {5e} - 3$</td><td>26.99 %</td><td>26.63 %</td><td>28.39 %</td><td>35.95 %</td><td>26.96 %</td><td>33.93 %</td></tr><tr><td>${LR} = {1e} - 3$</td><td>25.34 %</td><td>28.14 %</td><td>26.97 %</td><td>26.45 %</td><td>17.03 %</td><td>18.40 %</td></tr><tr><td>${LR} = {1e} - 2$</td><td>28.47 %</td><td>26.51 %</td><td>25.32 %</td><td>27.02 %</td><td>17.03 %</td><td>17.03 %</td></tr></table>

<table><tbody><tr><td rowspan="2"></td><td colspan="2">阿尔法 $= {0.01}$</td><td colspan="2">${Alpha} = {0.1}$</td><td colspan="2">阿尔法 $= {0.25}$</td></tr><tr><td>${BS} = {256}$</td><td>${BS} = {128}$</td><td>${BS} = {256}$</td><td>${BS} = {128}$</td><td>${BS} = {256}$</td><td>${BS} = {128}$</td></tr><tr><td>${LR} = {1e} - 4$</td><td>21.54 %</td><td>23.13 %</td><td>22.43 %</td><td>24.32 %</td><td>26.98 %</td><td>30.28 %</td></tr><tr><td>${LR} = {5e} - 3$</td><td>26.99 %</td><td>26.63 %</td><td>28.39 %</td><td>35.95 %</td><td>26.96 %</td><td>33.93 %</td></tr><tr><td>${LR} = {1e} - 3$</td><td>25.34 %</td><td>28.14 %</td><td>26.97 %</td><td>26.45 %</td><td>17.03 %</td><td>18.40 %</td></tr><tr><td>${LR} = {1e} - 2$</td><td>28.47 %</td><td>26.51 %</td><td>25.32 %</td><td>27.02 %</td><td>17.03 %</td><td>17.03 %</td></tr></tbody></table>

Table 5: Average of the validation results between cartoon and sketch domains.

表5:卡通与草图领域之间验证结果的平均值。

From the results we can notice that as alpha increases it is necessary to lower the learning rate to make the neural network learn without there being divergence in one of the losses. This can be seen with alpha $= {0.25}$ and high learning rates, since the average accuracy value is below 20%.

从结果中我们可以注意到，随着α的增加，有必要降低学习率，以使神经网络能够学习而不会出现其中一个损失发散的情况。这在α $= {0.25}$ 和高学习率的情况下可以看到，因为平均准确率值低于20%。

![bo_d2d0r6v7aajc738q0tqg_6_422_472_799_559_0.jpg](images/bo_d2d0r6v7aajc738q0tqg_6_422_472_799_559_0.jpg)

Figure 4: Example of hyperparameters that lead the network to perform bad on validation

图4:导致网络在验证中表现不佳的超参数示例

Finally, the best score is obtained with $\mathrm{{LR}} = 5\mathrm{e} - 3$ , alpha $= {0.1}$ and batch size $= {128}$ . Finally the test accuracy (i.e. on Art Painting domain) found with the best set of hyperparameters is 52.31 % (±1.72 %).

最后，使用 $\mathrm{{LR}} = 5\mathrm{e} - 3$、α $= {0.1}$ 和批量大小 $= {128}$ 获得了最佳分数。最后，使用最佳超参数集获得的测试准确率(即在艺术绘画领域)为52.31%(±1.72%)。

![bo_d2d0r6v7aajc738q0tqg_6_421_1321_793_538_0.jpg](images/bo_d2d0r6v7aajc738q0tqg_6_421_1321_793_538_0.jpg)

Figure 5: Plot of the losses obtained with the best hyperparameters set

图5:使用最佳超参数集获得的损失图

Note that for the simplicity of this report some results with high alpha values are omitted (e.g. alpha 0.5 and 0.8), since experimentally has been seen that the more alpha grew the worse the results are (because the discrimination loss tends to diverge after few epochs with high learning rates). For this reason further investigation has been done to reduce the number of epochs and consequently the step size (e.g. NUM EPOCHS = 10, STEP SIZE=6). Despite this effort, the model that performed better was the one proposed previously.

请注意，为了本报告的简洁性，一些α值较高的结果被省略了(例如α为0.5和0.8)，因为通过实验发现，α越大结果越差(因为在高学习率下经过几个epoch后，判别损失往往会发散)。因此，进行了进一步的研究以减少epoch的数量，从而减小步长(例如NUM EPOCHS = 10，STEP SIZE = 6)。尽管做出了这些努力，但表现更好的模型仍然是之前提出的那个。

## References

## 参考文献

[1] Krizhevsky, Alex & Sutskever, Ilya & Hinton, Geoffrey. (2012). ImageNet Classification with Deep Convolutional Neural Networks. Neural Information Processing Systems. 25. 10.1145/3065386.

[1] 克里兹夫斯基，亚历克斯 & 苏茨克维，伊利亚 & 辛顿，杰弗里。(2012年)。使用深度卷积神经网络进行ImageNet分类。神经信息处理系统。25。10.1145/3065386。

[2] Li, D., Yang, Y., Song, Y.Z., & Hospedales, T. (2017). Deeper, Broader and Artier Domain Generalization. In International Conference on Computer Vision.

[2] 李，D.，杨，Y.，宋，Y.Z.，& 霍斯佩代尔斯，T.(2017年)。更深、更广、更具艺术性的领域泛化。在国际计算机视觉会议上。

[3] Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., & Lempitsky, V.S. (2016). Domain-Adversarial Training of Neural Networks. ArXiv, abs/1505.07818.

[3] 加宁，Y.，乌斯季诺娃，E.，阿贾坎，H.，热尔曼，P.，拉罗谢尔，H.，拉维奥莱特，F.，马尔尚，M.，& 伦皮茨基，V.S.(2016年)。神经网络的领域对抗训练。ArXiv，abs/1505.07818。