# Medical Stable Diffusion

In this project, we aim to using the latest published **Stable Diffusion** to generate medical images, for example when we enter a prompt with disease, the medical image with that specific disease will be generated automatically.

## Stable Diffusion

Stable diffusion was the most popular open-source AI generating program in 2022. The basic idea behind using stable diffusion for image generation is to model an image as a random process that evolves over time. The process starts with a noisy image and the stable diffusion equation is used to evolve the image over time by adding noise to it. The stable diffusion equation used in image generation is a partial differential equation that describes how the image evolves over time. This equation incorporates a fractional Laplacian operator, which allows for long-range dependence in the image. The fractional Laplacian operator is a generalization of the standard Laplacian operator, which is used in traditional diffusion processes.

As the image evolves over time, it gradually becomes less noisy and more coherent. By controlling the parameters of the stable diffusion equation, we can control the rate of diffusion and the amount of noise added at each step, allowing us to generate a wide variety of images with different textures and structures. Stable diffusion-based image generation has shown promising results in generating naturalistic and diverse images, and has potential applications in computer vision, graphics, and art. 

## Dataset

We chose **CheXpert** to keep training the stable diffusion. It is a large dataset of chest X-rays and competition for automated chest x-ray interpretation, which features uncertainty labels and radiologist-labeled reference standard evaluation sets. Chest radiography is the most common imaging examination globally, critical for screening, diagnosis and management of many life threatening diseases. The dataset included 14 different kind of diseases in the label, with 1 and 0 in a csv file, which makes the prompt setting for the future training much easier.

## First Try with Normal Stable Diffusion

After combining the extracted features based on radiomics and the manual defined method, there were 126 features for one single case. Many of them actually didn’t have high relationship with the survival days, most of them were just noise in the final regression. We used the SpearmanR to show the relationship between the features and survival day. Set a threshold to choose the parameter and lower the outliers.

<div style="display: inline-block;">
    <img src="images/(a) No Finding.jpg" title="(a) No finding" width="250" height="auto"/>
    <img src="images/(b) Cardiomegaly.jpg" title="(b) Cardiomegaly" width="250" height="auto"/>
    <img src="images/(c) Pleural Effusion.jpg" title="(c) Pleural Effusion" width="250" height="auto"/>
</div>
  
## Regression Method

There are many papers have already proved that the simple machine learning regression methods like random forest or MLP have better performance compared with the deep learning methods. We do a comparison among all the normal regression methods and find that the pipeline which is combined with standard scaler and SGD regressor has the best performance among all of them.

<p align="center">
  <img src="./images/Regression_Methods.png" alt="regression methods" title="regression methods" width="400" height="auto">
</p>
  
## Framework of Survival

The whole framework of the survival prediction task looks like that:

<p align="center">
  <img src="./images/Framework.png" alt="framework" title="framework" width="700" height="auto">
</p>
  
## Results

We tried different threshold of the filter. It was clear that the threshold $>|0.10|$ showed the best result ($0.621$) among all of them. After filtering, the features of each case were reduced from 126 to 28 which also sped up the computation. The accuracy was far higher than another two related paper also extracted radiomic features in BraTS 2020.


|**Different Thresholds**|**Accuracy**|**MSE**|**SpearmanR**|
| --- | --- | --- | --- |
|without|0.379|$1.079\times10^{10}$|0.335|
|$>\|0.08\|$|0.586|$2.056\times10^8$|0.518|
|$>\|0.10\|$|**0.621**|$3.641\times10^7$|0.502|
|$>\|0.12\|$|0.517|$5.775\times10^7$|0.417|
|$<-0.01$|0.552|$5.167\times10^7$|0.480|
|$<-0.05$|0.379|$1.516\times10^8$|0.050|
|$<-0.10$|0.586|$1.406\times10^5$|0.537|

We also tried to use different input of the images since there were 4 types of images in the training dataset. It was also shown that the input of $T_1$ image had better result of accuracy compared with other images.

|**Different Input Modalities**|**Accuracy**|**MSE**|**SpearmanR**|
| --- | --- | --- | --- |
|$T_1$|**0.621**|$3.641\times10^7$|0.502|
|$T_{1ce}$|0.448|$2.977\times10^8$|0.433|
|$T_2$|0.483|$1.118\times10^5$|0.233|
|$T_{2Flair}$|0.345|$2.470\times10^5$|0.127|
|Combination of 4 modalities|0.517|$7.183\times10^6$|0.494|


## Conclusion

In the survival part, we achieved a promising accuracy ($0.621$). It was clear that the filter depended on the Spearman's Rank had an outstanding performance. The framework without the filter could only reach the accuracy of $0.379$ which was far lower. The reason why we always kept the negative part in the threshold was that the most import feature was age which showed $-0.419$ SpearmanR value with the survival days. And also due to some obvious common sense, such as tumor size and tumor surface area were all negatively correlated with survival time. According to the result, the accuracy with threshold $>|0.10|$ ($0.621$) was higher than the accuracy with threshold only $<-0.10$ ($0.586$), because some of features were also positively correlated with survival time like the tumor's distance and offset from the brain center. Setting the threshold is very important for a good result. It also needs researchers to have relevant medical knowledge of tumors and as we learn more medically about tumors, perhaps we can manually add more important parameters.
