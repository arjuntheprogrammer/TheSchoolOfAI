# TSAI_Session7

Project for The School of AI
---------------------------------------------------------------------------------------
## **Assignment7A :**

1. Check this paper out: https://arxiv.org/pdf/1409.4842.pdf (Links to an external site.)
2. They mention on page 6, that the RF is 224x224. 
3. Use the formulas for RF and show the calculations
4. Submit this in the readme file

### **Output Size and Receptive Field Calculation for Inception V**

[K=Kernal Size, S=Stride, P=Padding]

| SNo. | Input Size   | Layer/Operation | K | S | P | OutputSize   | Jin  | RF      |
| ---  | ---          | ---             |---|---|---| ---          | ---  | ---     |
| 1.   | 224X224X3    | Conv 3X3X64     | 7 | 2 | 3 | 112X112X64   | 1    | 7X7     |
| 2.   | 112X112X64   | MaxPooling      | 3 | 2 | 1 | 56X56X64     | 2    | 11X11   |
| 3.   | 56X56X64     | Conv 1X1X64     | 1 | 1 | 0 | 56X56X64     | 4    | 11X11   |
| 4.   | 56X56X64     | Conv 3X3X192    | 3 | 1 | 1 | 56X56X192    | 4    | 19X19   |
| 5.   | 56X56X192    | MaxPooling      | 3 | 2 | 1 | 28X28X192    | 4    | 27X27   |
| 6.   | 28X28X192    | Inception 3A    |   |   |   | 28X28X256    | 8    | 59X59   |
| 7.   | 28X28X256    | Inception 3B    |   |   |   | 28X28X480    | 8    | 91X91   |
| 8.   | 28X28X480    | MaxPooling      | 3 | 2 | 1 | 14X14X480    | 8    | 95X95   |
| 9.   | 14X14X512    | Inception 4A    |   |   |   | 14X14X512    | 16   | 159X159 |
| 10.  | 14X14X512    | Inception 4B    |   |   |   | 14X14X512    | 16   | 223X223 |
| 11.  | 14X14X512    | Inception 4C    |   |   |   | 14X14X512    | 16   | 287X287 |
| 12.  | 14X14X512    | Inception 4D    |   |   |   | 14X14X528    | 16   | 351X351 |
| 13.  | 14X14X528    | Inception 4E    |   |   |   | 14X14X832    | 16   | 451X451 |
| 14.  | 14X14X832    | MaxPooling      | 3 | 2 | 1 | 7X7X832      | 16   | 483X483 |
| 15.  | 7X7X832      | Inception 5A    |   |   |   | 7X7X832      | 32   | 611X611 |
| 16.  | 7X7X832      | Inception 5B    |   |   |   | 7X7X1024     | 32   | 739X739 |
| 17.  | 7X7X1024     | Average Pooling | 7 | 7 | 0 | 1X1X1024     | 224  |         |
| 18.  |              | Droupout        |   |   |   |              | 224  |         |
| 19.  |              | Linear          |   |   |   |              | 224  |         |
| 20.  |              | Softmax         |   |   |   |              | 224  |         |

P.S - for calculating the R.F. through the Inception block - I have considered the K=5 [maximum kernal size amoung other paralle kernels].

---------------------------------------------------------------------------------------

## **Assignment7B :**

1. Try to design the ENAS Network
2. Add skip connections to the network
3. Train for 100 Epochs (add BN and ReLU after every layer)
4. Submit the results.

### We are getting validation **accuracy of 70.26%** after 100 epochs for the Cifar 10 Dataset with the ENAS Network.

--------------------------------------------------------------------------------------
## Notebook Names: 
1. **Assignment7.ipynb**

--------------------------------------------------------------------------------------

## Model Architecture with Skip Connections
![model_plot](https://user-images.githubusercontent.com/15984084/64485579-78ec4200-d240-11e9-9b78-bf024628b8a6.png)

--------------------------------------------------------------------------------------



