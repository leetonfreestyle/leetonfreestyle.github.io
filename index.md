# 图像生成 Image Generation

在概率统计理论中, 生成模型是指能够随机生成观测数据的模型，尤其是在给定某些隐含参数的条件下。 它给观测值和标注数据序列指定一个联合概率分布。 在机器学习中，生成模型可以用来直接对数据建模（例如根据某个变量的概率密度函数进行数据采样），也可以用来建立变量间的条件概率分布。

## 按输入输出来区分

### noise -> image

> 通用图片生成(general image generation)

[《Goodfellow I, Pouget-Abadie J, Mirza M, et al. Generative adversarial nets[C]//Advances in neural information processing systems. 2014: 2672-2680.》](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

深度学习最成熟的部分是判别模型，给自然图像、语音信号、自然语言等高维数据标上类别标签，而生成模型由于其难以近似处理其中的概率问题而很少受到深度学习的青睐。

生成对抗网络提出了一个新的思路，使用生成器来生成图片，使用判别器来辨别这些图片的真伪，在训练过程中通过生成器与判别器的不断博弈来达到提升生成器生成效果的目标。

[《Arjovsky M, Chintala S, Bottou L. Wasserstein gan[J]. arXiv preprint arXiv:1701.07875, 2017.》](https://arxiv.org/pdf/1701.07875)

自从2014年Ian Goodfellow提出以来，GAN就存在着训练困难、生成器和判别器的loss无法指示训练进程、生成样本缺乏多样性等问题。

本文作者详细分析了训练GAN时的问题，发现了以下问题：
1. GAN定义的损失为真实分布与生成分布之间的距离，但由于这两个分布之间几乎不可能有不可忽略的重叠，因此判别器将轻易辨别且无法给生成器有效的梯度信息，导致训练过程中生成器梯度消失。
2. GAN定义的两种分布之间的距离不够好，对于“生成器宁可多生成一些重复但是很“安全”的样本，也不愿意去生成多样性的样本”的问题惩罚力度不够，导致模式坍塌（collapse mode）。

WGAN使用性能优良的Wasserstein距离的近似值作为损失进行拟合，既解决了训练不稳定的问题，也提供了一个可靠的训练进程指标，而且该指标确实与生成样本的质量高度相关。

[《Brock A, Donahue J, Simonyan K. Large scale gan training for high fidelity natural image synthesis[J]. arXiv preprint arXiv:1809.11096, 2018.》](https://arxiv.org/pdf/1809.11096)

文章的创新点是将正交正则化的思想引入 GAN，通过对输入先验分布 z 的适时截断大大提升了 GAN 的生成性能，在 ImageNet 数据集下 Inception Score 竟然比当前最好 GAN 模型 SAGAN 提高了 100 多分（接近 2 倍）。

> 基于先验信息的图像生成

[《Mirza M, Osindero S. Conditional generative adversarial nets[J]. arXiv preprint arXiv:1411.1784, 2014.》](https://arxiv.org/pdf/1411.1784)

在 CGAN 的工作中，假设y为额外输入的label信息，通过在输入层直接拼接样本与 y 信息的向量而实现先验信息的利用。具体使用的 y 信息有 one-hot vector，也有图像（也就是基于另一个图像去生成）。这个 y 信息的选择其实十分灵活，在后期的工作中也依然很常见，毕竟是一种非常直观有效的加入 label 信息的方式。

[《Yan X, Yang J, Sohn K, et al. Attribute2image: Conditional image generation from visual attributes[C]//European Conference on Computer Vision. Springer, Cham, 2016: 776-791.》](https://arxiv.org/pdf/1512.00570)

本文基于属性生成图像，降低了采样的不确定性，增加了生成图像的真实感。给定属性 y 和 latent variable z, 我们的目标是构建一个模型，可以在条件 y 和 z 的基础上产生真实的图像。

条件式图像产生是简单的两步操作：1. 随机的从先验分布 p(z) 中采样出 latent variable z; 2. 给定 y 和 z 作为条件变量，产生图像 x。 

### image -> image

> 图像翻译(image to image translation)

[《Isola P, Zhu J Y, Zhou T, et al. Image-to-image translation with conditional adversarial networks[J]. arXiv preprint, 2017.》](https://arxiv.org/pdf/1611.07004)

这篇文章主要提供了一个基于cGAN的模型，并且利用这个general的模型可以同时应用到多个任务场景中去，而不需要额外设置模型结果和目标函数。

[《Zhu J Y, Park T, Isola P, et al. Unpaired image-to-image translation using cycle-consistent adversarial networks[J]. arXiv preprint, 2017.》](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)

对于很多任务而言，配对训练数据可遇不可求。本文阐述了为什么要用循环的方式来扩展GAN，因为从A到B域映射出来的图片可能有非常多的可能，并且都满足B域的分布，加入一个反向映射的循环，可以加强转换的约束性，同时还能避免GAN中常见的mode collapse的问题，作者称其为cycle consistent。

[《Chen Q, Koltun V. Photographic image synthesis with cascaded refinement networks[C]//IEEE International Conference on Computer Vision (ICCV). 2017, 1(2): 3.》](http://openaccess.thecvf.com/content_ICCV_2017/papers/Chen_Photographic_Image_Synthesis_ICCV_2017_paper.pdf)

本文的目的是生成大尺寸的、质量接近照片的图像，现有的基于 GANs 的方法生成的图像在尺寸和逼真程度上都有各种问题，而 GANs 本身训练困难的特点更是雪上加霜。所以 GANs 的方法不适用。

为了同时达到全局协调、高分辨率、高记忆力三个特点，作者们设计了一个由多个分辨率倍增模块组成的级联优化网络 CRN。模型一开始生成的图像分辨率只有 4x8，通过串接的多个分辨率倍增前馈网络模块，分辨率逐步翻番，最终达到很高的图像分辨率（比如最后一个模块把512x1024的图像变成1024x2048）。

对于约束不完全的训练问题，作者借助一个 VGG-19 图像感知模型，提取它识别的图像特征中高低不同的某几层作为计算训练损失的依据，从而同时涵盖了图像特征中边缘、颜色等低级细粒度特征和物体、类别等高级总体布局特征，从而构建了全面、强力的损失函数。

> 图像超分辨率(image super resolution)

[《Dong C, Loy C C, He K, et al. Image super-resolution using deep convolutional networks[J]. IEEE transactions on pattern analysis and machine intelligence, 2016, 38(2): 295-307.》](https://arxiv.org/pdf/1501.00092)

SRCNN是首个使用CNN结构的端到端的超分辨率算法,令F为训练模型的目标函数，输入低分辨率图像Y，经过该函数的处理F(Y)，得到与高分辨率原图X尽可能相似的结果。网络结构实质上只有三层卷积，但已经取得不错的效果。

[《Ledig C, Theis L, Huszár F, et al. Photo-realistic single image super-resolution using a generative adversarial network[J]. arXiv preprint, 2017.》](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)

本文针对传统超分辨方法中存在的结果过于平滑的问题，提出了结合最新的对抗网络的方法，得到了不错的效果。并且针对此网络结构，构建了自己的感知损失函数。

### image + mask -> image

> 图像填补(image inpainting)

[《Pathak D, Krahenbuhl P, Donahue J, et al. Context encoders: Feature learning by inpainting[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2536-2544.》](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Pathak_Context_Encoders_Feature_CVPR_2016_paper.pdf)

本文的上下文编码器需要解决一个困难的任务:填补图像中大量缺失的区域，而这些区域无法从附近的像素中获得“提示”。

本文的主要思路为主要思路是结合Encoder-Decoder 网络结构和 GAN （Generative Adversarial Networks），Encoder-Decoder 阶段用于学习图像特征和生成图像待修补区域对应的预测图，GAN部分用于判断预测图来自训练集和预测集的可能性，当生成的预测图与GroundTruth在图像内容上达到一致，并且GAN的判别器无法判断预测图是否来自训练集或预测集时，就认为网络模型参数达到了最优状态。

[《Nazeri K, Ng E, Joseph T, et al. EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning[J]. arXiv preprint arXiv:1901.00212, 2019.》](https://arxiv.org/pdf/1901.00212)

本文提出了一个二阶段生成对抗网络 EdgeConnect，它包括一个边缘生成器，然后是一个图像补全网络。边缘生成器在图像的缺失区域（规则和不规则）生成预测边缘，然后图像补全网络使用预测边缘作为先验填充缺失区域。

### image + image -> image

> 风格迁移(Neural Style)

[《Gatys L A, Ecker A S, Bethge M. A neural algorithm of artistic style[J]. arXiv preprint arXiv:1508.06576, 2015.》](https://arxiv.org/pdf/1508.06576)

本文利用深度学习的方法来重建纹理，解决了手动建模局部统计模型描述生成纹理的困扰。作者通过两个不同的网络分别提取纹理和不包括风格的图像内容，然后再合成目标图像。

[《Liao J, Yao Y, Yuan L, et al. Visual attribute transfer through deep image analogy[J]. arXiv preprint arXiv:1705.01088, 2017.》](https://arxiv.org/pdf/1705.01088)

论文提出了一种新的两张图片直接进行视觉属性迁移的方法。该方法针对的是两张具有不同内容却有相似语义的图像，比如两张图的主体是同一种类别的物体，并利用高层抽象特征建立起了两张图的内容的语义对应关系。 

这种图像视觉属性迁移方法可以在结构上基本保留两张图中内容图的内容及结构，同时融入参考图的视觉属性。和之前的算法不同的是，这种方法甚至适用于输入是真实照片，输出也希望是真实照片的任务，即可以达到像素级别的迁移。

## 参考文献

1. [Presentations of Ian Goodfellow](http://www.iangoodfellow.com/slides/)
1. [CVPR 2018 Tutorial on GANs](https://sites.google.com/view/cvpr2018tutorialongans/)
1. [开发者自述：我是这样学习 GAN 的](https://www.leiphone.com/news/201707/1JEkcUZI1leAFq5L.html)
1. [深度解读DeepMind新作：史上最强GAN图像生成器—BigGAN](https://www.jiqizhixin.com/articles/2018-10-12-9)
1. [谱归一化（Spectral Normalization）的理解](https://blog.csdn.net/StreamRock/article/details/83590347)
1. [深度学习中的Lipschitz约束：泛化与生成模型](https://www.jiqizhixin.com/articles/2018-10-16-19)
1. [令人拍案叫绝的Wasserstein GAN](https://zhuanlan.zhihu.com/p/25071913)
1. [如此逼真的高清图像居然是端到端网络生成的？GANs 自叹不如 ICCV 2017](https://www.leiphone.com/news/201708/Jy2RophpB7M9WIhf.html)
1. [NTIRE 2018 超分辨率 CVPR Workshop](https://zhuanlan.zhihu.com/p/39930043)
1. [结合深度学习的图像修复怎么实现？](https://www.zhihu.com/question/56801298)
1. [图像风格迁移(Neural Style)简史](https://zhuanlan.zhihu.com/p/26746283)
1. [学习笔记：图像风格迁移](https://blog.csdn.net/czp_374/article/details/81185603)
1. [探索生成式对抗网络GAN训练的技术：自注意力和光谱标准化](https://cloud.tencent.com/developer/article/1346708)
