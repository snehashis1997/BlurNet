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

<p align="center">
  <img src="https://user-images.githubusercontent.com/33135767/93631709-b06ffb80-fa09-11ea-8b3c-db101cf51a33.gif"/>
</p>
