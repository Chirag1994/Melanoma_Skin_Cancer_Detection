# Melanoma_Skin_Cancer_Detection

---

Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective. Currently, dermatologists evaluate every one of a patient's moles to identify outlier lesions or “ugly ducklings” that are most likely to be melanoma. Melanoma is a deadly disease, but if caught early, most melanomas can be cured with minor surgery.

In this project, we have made an attempt to create a web application that dermatologists could use to enhance the diagnostic accuracy given the images of patient's skin images to determine which images represent melanoma.

To build the Web Application, we are leveraging the dataset of the competition organized on [Kaggle](https://www.kaggle.com/competitions/siim-isic-melanoma-classification).

The dataset can be downloaded from [here](https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-512x512).

### Repository Structure

---

The project has the following structure:

- Scripts: This Repository contains the training, validation, inferencing, helper utilities etc. python modules (`.py files`).
- notebooks: This Repository contains the notebooks used to build the model pipeline. It contains 3 notebooks, one that used only images, one that used noth images & tabular features and the last one named as `melanoma-final-model-kaggle.ipynb` which has in-depth steps right from training the model pipeline to building a Web Application and hosting to `Hugging Face Spaces.`
- input: This repository contains the training, testing images and tabular features (`provided in csv files`). It only contains only a few images which can be downloaded from the above link.
- output: This Repository contains the binary/model trained file that will be used for predicting on new images, right now it is empty.

`NOTE`: To build the model pipeline, we have used Kaggle computing resources, specifically `multi-gpu's offered by kaggle`, the free version of Google Colab crashed while training the model.

### Deployment

---

The Web Application is currently deployed at Hugging Face Spaces [here](https://huggingface.co/spaces/Chirag1994/Melanoma_Skin_Cancer_Detection_App/tree/main).
