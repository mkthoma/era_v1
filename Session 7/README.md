# Designing a CNN for digit classification on MNIST data using PyTorch
## Introduction
The aim of this exercise is to design a efficient CNN model which is capable of identifying handwritten digits in the MNIST dataset. The design of the neural network will be based on a few parameters such as:
- 99.4% accuracy
- Less than or equal to 15 Epochs
- Less than 8000 Parameters
- Using modular code 

## Designing the Models
We will dive deeper into how we can deisgn an architecture that can give us the expected results while keeping in line with the contraints stated above. 
### Step 1
-  `model1`

    - Target - First lets look at a basic skeleton that can provide us a foundation to work with. Here we will not be sticking to any of the constraints stated above. We will just be looking at the accuracy of the model and understand whether the model is overfitting or whether it can be used further for our model development. We shall define the dataloader, a working code, a skeleton for our neural network, training and testing less with optimizer for the training part and displaying the basic data that we are working with. 
   - Results 
       
        - Total parameters - 19,990
        - Best Training Accuracy - 99.27%
        - Best Testing Accuracy - 98.91% 
    - Analysis - The model is light since it only has 20k parameters. The model is also performing overall pretty well since the overfitting it very minimal. We shall work on reducing the number of parameters and the epochs in the next model.

- `model2`

  - Target - Using the same model as before we will try to reduce the number of paramters by reducing the kernel size and stick to out limit of 15 epochs. We will try to see how the accuracy for this mdoel will change and decide whether we can use the same skeleton for developing our model on. 
  - Results
    
      - Total parameters - 16,040
      - Best Training Accuracy - 98.98%
      - Best Testing Accuracy - 98.67% 
  - Analysis -  The total parameters have been reduced by around 4k and the model is still performing pretty well. The model is still overfitting but the margin is very small. The epochs have been reduced and still the model is giving a good accuracy without batch normalization and dropouts. The model is good for us to build a skeleton around it and see if we can start to bring in the paramter contraints to this model and see the final accuracy.

- `model3`
  
  - Target - We will look into how we can reduce the number of parameters in this model. We will do this by decreasing the kernel size of each convolution blocks without any other major changes and see how the model works.
  - Results
    
      - Total parameters - 7,092
      - Best Training Accuracy - 98.83%
      - Best Testing Accuracy - 98.38% 
  - Analysis -  Great news! Our total paramters has been decreased by half but the model accuracy still holds. The issue of overfitting still remains but it is negligible. Now we are sure we can proceed with this particular model. 

### Step 2
- `model4` model5 in colab
  - Target - We will look into how we can increase the efficiency of the model. For this first we shall add batch normalization to each kernel and see the change in parameters and accuracy.
  - Results
    
      - Total parameters - 7,200
      - Best Training Accuracy - 99.49%
      - Best Testing Accuracy - 99.23% 
  - Analysis -  Our total paramters has increased by a small amount (100) but the accuracy of the model has gone up better than before. By introducting batch normalization we have increased the efficiency of the model.
- `model5` 
  - Target - Now that we have added batch normalization, we shall add dropout to each layer in the nerual network. We are setting the dropout value to 0.1 i.e, 10% dropout. 
  - Results
    
      - Total parameters - 7,200
      - Best Training Accuracy - 98.64%
      - Best Testing Accuracy - 99.14% 
  - Analysis -  Our total paramters has remained the same but the accuracy of the model has gone up better than before. We can see that the model is underfitting now and the overfitting is not present anymore. This is great news for our model.
- `model6` 
  - Target - Now that we have added batch normalization and dropout , we shall remove the heavy kernel that is defined towards the end and replace it with a Global Average Pooling and see how it affects the model. 
  - Results
    
      - Total parameters - 2,300
      - Best Training Accuracy - 96.24%
      - Best Testing Accuracy - 95.73% 
  - Analysis -  Our total paramters has decreased drastically to about one-fourth the total paramters in the previous model. With such a drop in paramters, it is expected that the accuracy will be less and the numbers clearly reflect that. In our next model we shall try to increase the parameters to the original number of 7k we saw in the earlier model and see how it performs by increasing the capacity.
- `model7` 
  - Target - In order to restore the original paramter count of around 7k, we shall increase the capacity of this model containing Global Average Pooling(GAP) instead of a heavy 7x7 kernel. We will increase the kernel size and add a 1x1 transition block after the GAP.
  - Results
    
      - Total parameters - 7,550
      - Best Training Accuracy - 98.92%
      - Best Testing Accuracy - 99.4% (14th and 15th epoch)
  - Analysis -  On increasing the capacity we have observed that the model is outperforming all the previous models and there is no overfitting but there is underfitting. We have almost met all our objectives in terms of less than 8k parameters, 15 epochs, and 99.4% accuracy. We would like to see the model producing the accuracy more consistently so we shall try image augmentation and learning rate schedulers to see how much the model can be pushed further.
### Step 3
- `model8` 
  - Target - In order to consistenly have our mdoel produce an accuracy of 99.4% or above, we shall try to introduce image augmentation and learning rate schedulers in this new model.
  - Results
    
      - Total parameters - 7,550
      - Best Training Accuracy - 98.82%
      - Best Testing Accuracy - 99.4% (12th to 15th epoch)
  - Analysis -  We have augmented the image using random rotation between -7 and 7 degrees and introduced a learning rate on plateau in this model. The model is performing better than before and able to achieve a accuracy of 99.41% twice and 99.38%. Different learning rate using step and one cycle were tried out but this one has the best accuracy observed. It is better to let the model decide how the learning rate should be changed rather than us trying to make the choice for the model. 
