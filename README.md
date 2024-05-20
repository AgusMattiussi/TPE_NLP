# TPE NLP

## Installation 
**Create Python virtual environment and install the dependencies.**

**LINUX:**
```
python3.7 -m venv .nlp_venv #create it
source .nlp_venv/bin/activate #activate it
```

**Install Requirements**
```
pip install -r requirements.txt
```

**English stopwords**
```
import nltk
nltk.download('stopwords')
```
And mannually added some words as needed ('also', 'said', 'Mr', 'Mrs', 'would', 'will', 'one', 'two', 'three', 'four', 'new', 'like', 'way', 'get', 'say')