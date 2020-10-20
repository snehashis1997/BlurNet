## Design a CNN based deep learning model which classify both blurred and not blurred version of a data set where atleast one class is extremely rare i.e it is a highly           unbalanced   dataset and deploy it in a web app

## DataSet Generation: 

1) Take any publicly available unbalanced Image dataset with atleast 8-10 categories in the 
   target variable  

2)  Make the dataset extremely skewed (.i,e frequently occurring categories should contribute to around 80-90% od total observations). There must be atleast one category which is     a rare category( .i.e less than 3% of total observations fall under this category) 

3)  Now use any CV algorithms to reduce the Sharpness in the Image(feel free to use any other methods too) and make them blurry - atleast 30% of images in each category should be     blurred 

4) Using Flask and Ngrok Deploy it as a webapp

## Model Generation: 

1)  Use any model of your choice to make predictions 

2)  Conduct an analysis of how the accuracy of the model is varying on the Frequently occurring categories with the non frequent categories (rare) 

3)  Conduct an analysis on how the model is performing on the non blurry images and blurry images across all categories 

# My approch to solve this problem: 

* Define a function which can detect whether an image is blurry or not. I used [Laplacian filter] to detect whether the image is blurry or not with a threshold value 100. Generally blurry images laplacian filer values are too much low below 10 and not blurry values are way bigger like 1000.

### For Edge detection part 
- First detect whether the image is blurry or not.
- If it is not blurry then it use Canny edge filerting directly upon it. 
- But it is blurry then we can not use canny edge filtering directly because Canny edge filter take the image then first use a gaussian kernel to blur the image. So, first step   is already done for this steps. So I implement other steps as functions which are

   * Non maximum supression
   * Thresholding
   * Hystersis thresholding

## Not Bluried version of the avobe image and it's detected edges
![image](https://user-images.githubusercontent.com/33135767/96608711-553a6d00-1317-11eb-8ada-31f989543536.png) ![image](https://user-images.githubusercontent.com/33135767/96608743-5ff50200-1317-11eb-9692-049fe7d0b4b3.png)


## Bluried version of the avobe image and it's detected edges
![image](https://user-images.githubusercontent.com/33135767/96608782-6b482d80-1317-11eb-8c51-eaccc8489afc.png) ![image](https://user-images.githubusercontent.com/33135767/96608831-77cc8600-1317-11eb-8051-6750609248f4.png)


## Count plot before and after making classes highly imbalanced
![image](https://user-images.githubusercontent.com/33135767/96608605-2fad6380-1317-11eb-98a7-5f71ea1d7ce0.png) ![image](https://user-images.githubusercontent.com/33135767/96608643-3cca5280-1317-11eb-9943-a35e146730ec.png)

## My model architecture
<p align="center">
  <img src="https://user-images.githubusercontent.com/33135767/96608937-90d53700-1317-11eb-8776-1ae32d74f6ef.png"/>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/33135767/93631709-b06ffb80-fa09-11ea-8b3c-db101cf51a33.gif"/>
</p>

[Laplacian filter]: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

