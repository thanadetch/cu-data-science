# Research Multi Label Classification BERT

### Description

In the world of academic research, the categorization and classification of literature play a vital role in literary work. To ensure that the scholarly pieces are discoverable, accessible, and essentially relevant to their audience, the classification of literature requires to be proposed with precision and robustness. With databases like Scopus housing millions of articles, achieving high effectiveness and accuracy via human interpretation is much more challenging and labour-intensive. One of the promising solutions is multi-label text classification.

Your task is to multi-label classify multi-content text on the limited Scopus dataset into 18 subject areas of engineering: civil, environmental, biomedical, petroleum, metallurgical, mechanical, electrical, computer, optical, nano, chemical, materials, agricultural, education, industrial, safety, "mathematics and statistics", and material science.

The provided data comprises the papers' abstracts, titles, and labels, so have fun and good luck.

Here’s the text from the image converted to Markdown:

---

### Evaluation

The evaluation matrix for public and private ranking is macro F1-score, which is described as follows:

$$
F1_{macro} = \frac{1}{N} \sum_{i=1}^{N} F1_i
$$

Where `N` is the number of classes.

For each class, treat it as the positive class and group all the other classes as the negative class. Calculate the F1-score for this binary classification.

$$
F1_i = 2 \times \frac{Precision_i \times Recall_i}{Precision_i + Recall_i}
$$

#### Precision

The ratio of correctly predicted positive observations to the total predicted positives.

$$
Precision_i = \frac{TP_i}{TP_i + FP_i}
$$

#### Recall (Sensitivity or True Positive Rate)

The ratio of correctly predicted positive observations to all the actual positives.

$$
Recall_i = \frac{TP_i}{TP_i + FN_i}
$$

---

### Submission File
For each "id" in the test set, you must predict a one hot vector for the TARGET variable. The file should contain a header and have the following format:

```csv
 id, CE,ENV,BME,PE,METAL,ME,EE,CPE,OPTIC,NANO,CHE,MATENG,AGRI,EDU,IE,SAFETY,MATH,MATSCI
 001eval,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
 002eval,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0
 003eval,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0
 004eval,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0
 005eval,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0
 006eval,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0
 etc.
```

---

### Files

- `train_for_student.json` - the training set
- `test_for_student.json` - the test set

### Data Structure

- **Abstract**
- **Title**
- **Classes**

---

### Multi-labels of Subject Areas (`Classes`)

- **CE** - Civil Engineering
- **ENV** - Environmental Engineering
- **BME** - Biomedical Engineering
- **PE** - Petroleum Engineering
- **METAL** - Metallurgical Engineering
- **ME** - Mechanical Engineering
- **EE** - Electrical Engineering
- **CPE** - Computer Engineering
- **OPTIC** - Optical Engineering
- **NANO** - Nano Engineering
- **CHE** - Chemical Engineering
- **MATENG** - Materials Engineering
- **AGRI** - Agricultural Engineering
- **EDU** - Education
- **IE** - Industrial Engineering
- **SAFETY** - Safety Engineering
- **MATH** - Mathematics and Statistics
- **MATSCI** - Material Science
