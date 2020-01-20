# TSAI Session 20

Project for The School of AI
---------------------------------------------------------------------------------------

# Objectives
1. Refer to the code mentioned on pages: 182-195 of this BOOK http://faculty.neu.edu.cn/yury/AAI/Textbook/Deep%20Learning%20with%20Python.pdf

2. Repeat the same code while adding code comments. 
3. Train the GLOVE based model with 8000 samples instead of 200. 
4. Submit on GitHub and share the link. 
5. Mention your results along with your training and validation charts on the ReadMe page.

--------------------------------------------------------------------------------------

**Word Embedding** => Collective term for models that learned to map a set of words or phrases in a vocabulary to vectors of numerical values.

--------------------------------------------------------------------------------------

### Notebook Names: 
1. **Assignment20.ipynb**

--------------------------------------------------------------------------------------
## Training the model without pretrained word embeddings with 10000 samples

* Training Loss: 31.09% , Training Accuracy: 87.03%
* Validation Loss: 52.02% , Validation Accuracy: 75.52%

![acc3](https://user-images.githubusercontent.com/15984084/72747009-8398e980-3bd9-11ea-895e-9ac14c3190e6.png)
![loss3](https://user-images.githubusercontent.com/15984084/72747010-84318000-3bd9-11ea-90a9-846046a96ba8.png)

--------------------------------------------------------------------------------------
## GLOVE based model with 100 samples
* Training Loss: 27.17% , Training Accuracy: 89.50%
* Validation Loss: 90.34% , Validation Accuracy: 50.13%

![loss1](https://user-images.githubusercontent.com/15984084/72746545-64e62300-3bd8-11ea-9ee3-eaa592cd16ce.png)

![acc1](https://user-images.githubusercontent.com/15984084/72746546-64e62300-3bd8-11ea-90f3-9536022c07da.png)

--------------------------------------------------------------------------------------

## GLOVE based model with 8000 samples
* Training Loss: 23.34% , Training Accuracy: 91.29
* Validation Loss: 146.85% , Validation Accuracy: 49.39%

![loss2](https://user-images.githubusercontent.com/15984084/72746547-657eb980-3bd8-11ea-881a-2e42f7eeab49.png)

![acc2](https://user-images.githubusercontent.com/15984084/72746548-657eb980-3bd8-11ea-97cb-d397111b1d52.png)

--------------------------------------------------------------------------------------
