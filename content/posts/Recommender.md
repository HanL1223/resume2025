---
title: "Novel Recommender"
description: "Beginning to build a recommendation system"
dateString: May 2023
draft: false
tags: ["Python","Recommender", "Machine Learning","Collaborative Filtering","Emsemble Machine Learning"]
showToc: false
weight: 204
cover:
    image: "projects/recommender/cover.jpg"
--- 
### Github link
🔗 [Code](https://github.com/HanL1223/novel_recommender)
### Credit
This notebook is inspire by [Paper ](https://arxiv.org/pdf/2008.01192.pdf)
### Skill invlove

**Python**  **Collaborative Filtering**  **Data preprocessing**  **ScikitLearn** 


## Project Description
This project aims to build a recommender system for book ratings. The system utilizes collaboration filtering methods and ensemble techniques to provide accurate recommendations to users based on their preferences. The project includes data exploration, preprocessing, model selection, model construction, and evaluation phases.

The recommender system employs three standalone models, implementing collaboration filtering methods. The models are fine-tuned and compared using cross-validation to select the best-performing model. The chosen model is then further optimized using random search cross-validation. The top-performing models, including SVD, SVDpp, and KNNwithZscore, are used to construct an ensemble model for testing.

## Project Takeaways
Throughout the development of the Recommender System for Book Ratings, I gained several key takeaways:

- Enhanced Python skills, especially in data manipulation, model training, and evaluation.
- Deepened understanding of collaborative filtering techniques and their application in recommender systems.
- Explored data exploration techniques to identify trends and patterns in book ratings.
- Learned data preprocessing methods to improve data quality and reduce training time.
- Experienced model selection processes using cross-validation to identify the best-performing model.
- Practiced fine-tuning models with random search cross-validation for improved accuracy and robustness.
- Constructed an ensemble model by combining the top-performing models, showcasing the benefits of leveraging multiple models.
- Utilized evaluation metrics (MAE and RMSE) to quantitatively measure the performance of the recommender system.
- Identified limitations of collaborative filtering, such as cold-start issues and the presence of unseen books.
- Recognized the potential for future enhancements, including incorporating content-based methods and hybrid approaches.
- Considered scalability using frameworks like Spark to handle larger datasets and improve system performance.(Future improvement)

## Usage
The recommender system can be used to provide book recommendations based on user preferences. Users can input their user ID or book name to receive personalized recommendations. The system utilizes collaborative filtering techniques and an ensemble model to generate accurate recommendations.

To obtain recommendations for a specific user:
```python
import recommender_system
book_name = "Angels & Demons (Robert Langdon, #1)"  # Name of the book for which recommendations are required
recommendations(df = test_data,book_name = book_name)

id user_id	rating
33038	1783	4
26076	1483	4
193	6	3
26074	1541	3
930	65	3
```
For sample output ,we look at the reading history of top user 1783
```python
train[train['user_id'] == 1483]
```
| user_id | item_id | rating | book_name |                                                   |
| ------: | ------: | -----: | --------: | ------------------------------------------------- |
|  135608 |    1483 |    399 |         4 | The Da Vinci Code (Robert Langdon, #2)            |
|  135673 |    1483 |    456 |         4 | Memoirs of a Geisha                               |
|  135757 |    1483 |   1074 |         5 | 1984                                              |
|  135885 |    1483 |   1113 |         4 | Harry Potter and the Order of the Phoenix (Har... |
|  136250 |    1483 |   1302 |         5 | The Devil in the White City                       |

For the 1st place recommendation, since this user readed the Da Vinci Code, they might also interesting in its prequel. So the recommendation logically make sence

