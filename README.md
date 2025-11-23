# How to use

### #1  
git clone https://github.com/jonaslefdal/RecommenderSystem.git<br>
    cd RecommenderSystem

### #2 
WINDOWS:<br>
    python -m venv venv<br>
    .\venv\Scripts\activate<br><br>
MAC: <br> 
    python -m venv venv <br>
    source venv/bin/activate

### #3
pip install -r requirements.txt

### #4 
python src/app.py


<h2>Offline Evaluation</h2>

Run the Jupiter Notebook

<h2>Dataset</h2>

### 1 Dataset download
Download https://www.kaggle.com/datasets/najzeko/steam-reviews-2021
<br>You need to either download the full dataset (all columns), or choose: "author.steamid", "app_id", "app_name", "recommended"

### 2 File location
Place full Dataset in RecommenderSystem/data/

### 3 Run convert script on the dataset 
python convertFile.py<br>
You should get a "steam_reviews_trimmed.csv" in the /data folder
