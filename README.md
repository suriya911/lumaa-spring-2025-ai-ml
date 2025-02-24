# 🎬 AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

This project implements a **content-based recommendation system** that suggests movies based on a user-provided text description. It uses **TF-IDF vectorization** and **cosine similarity** to identify the most relevant movies from a small, fixed dataset.

## 🚀 How It Works

1. **User Input**: The user provides a short text description of their movie preferences.
2. **Dataset Loading**: A fixed CSV file (`dataset.csv`) containing movie details and plot descriptions is loaded.
3. **Text Preprocessing**: Missing plot summaries in the dataset are replaced with empty strings.
4. **Vectorization**: Both the movie plots and the user input are transformed into TF-IDF vectors.
5. **Similarity Computation**: The **cosine similarity** between the user query vector and each movie's TF-IDF vector is calculated.
6. **Recommendation**: The system returns the movie with the highest similarity score as the top recommendation, along with additional similar movies if needed.

## 🛠 Setup & Installation

### 1. Clone the Repository

To get started, clone the repository to your local machine using:

```bash
git clone https://github.com/suriya911/lumaa-spring-2025-ai-ml
cd lumaa-spring-2025-ai-ml
```

### 2. Create a virtual environment

It is recommended to use a virtual environment to keep dependencies isolated.

- For macOS/Linux:

  ```
  python3 -m venv <env_name>
  source rec_env/bin/activate
  ```

- For Windows (PowerShell):

  ```
  python -m venv <env_name>
  rec_env\Scripts\activate
  ```

### 3. Install Dependencies

- Ensure you have **Python 3.10** installed. Install all required dependencies using:

  ```bash
  pip install -r requirements.txt
  ```

- Open the file `recomender.py`

## 📊 Example Input and Output

### Input:

![Input Image](Pics\1.png)

### Output:

![Output Image](Pics\2.png)

## 🎥 Demo

A short screen recording demonstrating the recommendation system can be found here:  
[📹 Demo Video](https://drive.google.com/drive/folders/11PLhNxO_XMgu0RjuEpTnhcXLZ92gOewp?usp=sharing)

## 📝 Author Information

- Name: Suriya Chellappan
- Email: suriya.chellappan@sjsu.edu
- LinkedIn: https://www.linkedin.com/in/suriya-chellappan/
