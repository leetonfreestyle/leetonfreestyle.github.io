# 图像生成 Image Generation

在概率统计理论中, 生成模型是指能够随机生成观测数据的模型，尤其是在给定某些隐含参数的条件下。 它给观测值和标注数据序列指定一个联合概率分布。 在机器学习中，生成模型可以用来直接对数据建模（例如根据某个变量的概率密度函数进行数据采样），也可以用来建立变量间的条件概率分布。

## 按输入输出来区分

### noise -> image

> 通用图片生成(general image generation)

[《Goodfellow I, Pouget-Abadie J, Mirza M, et al. Generative adversarial nets[C]//Advances in neural information processing systems. 2014: 2672-2680.》](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

[《Arjovsky M, Chintala S, Bottou L. Wasserstein gan[J]. arXiv preprint arXiv:1701.07875, 2017.》](https://arxiv.org/pdf/1701.07875)

[《Brock A, Donahue J, Simonyan K. Large scale gan training for high fidelity natural image synthesis[J]. arXiv preprint arXiv:1809.11096, 2018.》](https://arxiv.org/pdf/1809.11096)

### image -> image

> 图像翻译(image to image translation)

[《Isola P, Zhu J Y, Zhou T, et al. Image-to-image translation with conditional adversarial networks[J]. arXiv preprint, 2017.》](https://arxiv.org/pdf/1611.07004)

> 图像超分辨率(image super resolution)

[《Dong C, Loy C C, He K, et al. Image super-resolution using deep convolutional networks[J]. IEEE transactions on pattern analysis and machine intelligence, 2016, 38(2): 295-307.》](https://arxiv.org/pdf/1501.00092)

### image + mask -> image

> 图像填补(image inpainting)

[《Pathak D, Krahenbuhl P, Donahue J, et al. Context encoders: Feature learning by inpainting[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2536-2544.》](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Pathak_Context_Encoders_Feature_CVPR_2016_paper.pdf)

### image + image -> image

> 风格迁移(image style transfer)

[《Gatys L A, Ecker A S, Bethge M. A neural algorithm of artistic style[J]. arXiv preprint arXiv:1508.06576, 2015.》](https://arxiv.org/pdf/1508.06576)

## 其他

### [Presentations of Ian Goodfellow](http://www.iangoodfellow.com/slides/)

### [CVPR 2018 Tutorial on GANs](https://sites.google.com/view/cvpr2018tutorialongans/)

### [开发者自述：我是这样学习 GAN 的](https://www.leiphone.com/news/201707/1JEkcUZI1leAFq5L.html)

### [深度解读DeepMind新作：史上最强GAN图像生成器—BigGAN](https://www.jiqizhixin.com/articles/2018-10-12-9)

### [谱归一化（Spectral Normalization）的理解](https://blog.csdn.net/StreamRock/article/details/83590347)

### [令人拍案叫绝的Wasserstein GAN](https://zhuanlan.zhihu.com/p/25071913)

### [NTIRE 2018 超分辨率 CVPR Workshop](https://zhuanlan.zhihu.com/p/39930043)

### [结合深度学习的图像修复怎么实现？](https://www.zhihu.com/question/56801298)
