# PGOpi

Please note that this is research code. It is not optimized, thus it can be slow, not very clean, and not well documented.

## Before proceeding download experiments data available at:

- [DATA](https://drive.google.com/file/d/15NVoqQO6EnK6t2nn887k7IY1kqInNjn4/view?usp=sharing): place it into the folder `datasets/`. 
- [EMBEDDINGS](https://drive.google.com/file/d/1_38NjvjhqkIxEI8IqP9xWBFA9PexGmpi/view?usp=sharing): place it into the folder `datasets/w2v/`. 
- [PAPER RESULTS](https://drive.google.com/file/d/1_mtbWTYHBvruZ85WumgKQ4LsYqpJ0tZk/view?usp=sharing): place it into the folder `experiments/`. 


### 1 - Build Preprocess Test Data for all classes
> python build_tests.py

### 2 - Build Preprocessed Training Data for each product class and each threshold
> python build_training.py 

### 3 - Fit each model for each product class and threshold
> python fit_co-att.py

### 4 - Run Repetitions for each best model configuration
> python rep-co-att.py

### **Dependencies**
RUN *requirements.txt* for solving dependencies
