## What's this?
Sharing the data is one of the non-technical problems of Machine Learning and Deep Learning projects.
On EMWProject, we are sharing only the URLs of the articles and the script to extract the texts from URLs.
However, some ofthe URLs can be expired or the content can be moved to another URL.
In this work, we are focusing on only one of the tasks in CLEF 2019. The task we are focusing is a classification problem.
Participants are asked to build a classifier thatwill classify the news whether they are related to a protest event or not.
The dataset for this task consists of  **3500 news**, **2500 non-protest** and  **900 protest news**.
The questions are;

* **Do we need to share all this data?**
* **Can we change the ratio of the news?**

Due to this reason, we did some experiments on our data set with **classical and advanced machine learning algorithms** to see how 
the missing URLs can affect the results of the models. The results of these experimentscan help us to simplify the data set and resolve 
the copyright issues. On our previous work, we were asked to build protest-classifier. Different algorithms were tried and due to propertiesof 
the shared dataset, which is not the same with CLEF 2019 dataset, classical machine learning algorithms such assupport vector machine, 
did better than advanced algorithms such as multilayer neural network, *BERT*.Hence, we will do our experiments with four different algorithms 
which are *Naïve Bayes*, *Support Vector Machine*, *Multi-Layer NN* and *BERT*. The complexity of algorithms is increasing, respectively.To sum up, 
we are working on a text classification problem. *TfIdfVectorizer()* method of *scikit-learn* library is used toprepare the dataset for the algorithms, except *Bi-LSTM* and *BERT*.

***
Due to copyright issues, we are not allowed to share the dataset. However, if you are interested on the work and dataset feel free to contact.
***

## How to use?

On 
