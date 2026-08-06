# Online Retail Dashboard

Interactive Streamlit dashboard for the UCI "Online Retail" dataset — sales
trends, top products, geography, customer purchase behaviour, and a
Linear Regression model predicting customer spend from Recency & Frequency.

## 1. Run it locally

```bash
# from inside this folder
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

It will open at http://localhost:8501.

**Getting data into it** — two options:
- Upload your `Online_Retail.xlsx` (or a cleaned CSV) via the file uploader
  in the sidebar when the app is running, **or**
- Drop the file into `data/Online_Retail.xlsx` before starting the app —
  it will be picked up automatically if nothing is uploaded.

## 2. Deploy for free on Streamlit Community Cloud

1. Create a **public** GitHub repo and push this whole folder to it:
   ```bash
   git init
   git add .
   git commit -m "Initial dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick your repo/branch, and set the main file path to
   `app.py`.
4. Click **Deploy**. You'll get a public URL like
   `https://<your-app-name>.streamlit.app`.

### A note on the data file
The raw `Online_Retail.xlsx` file is fairly large (~23MB) and, more
importantly, contains data you may not want to publish. Two good options:

- **Recommended:** don't commit the raw file at all (the `.gitignore` here
  already excludes it). Instead, rely on the in-app **file uploader** 

- **Alternative:** export your already-cleaned dataframe to a smaller CSV
  (`df_clean.to_csv("data/online_retail_clean.csv", index=False)`) and commit
  *that* instead — it's smaller and skips re-running the cleaning pipeline on
  every load. If you do this, remove the exclusion line for that specific
  file from `.gitignore`.

## 3. Project structure

```
retail_dashboard/
├── app.py              # the dashboard
├── requirements.txt    # pinned dependencies
├── .gitignore
├── README.md
└── data/               # put your data file here (not committed by default)
```

## 4. My thinking for prod 

- **Caching** (`@st.cache_data`) — the cleaning pipeline and model training
  only re-run when the underlying data actually changes, not on every UI click.
- **Error handling** — bad/missing files produce a clear on-screen message
  instead of a crash with a raw Python traceback.
- **Reproducible dependencies** — `requirements.txt` pins exact versions so
  the deployed app behaves the same as the local one.
- **Separation of data and code** — the app never assumes a hardcoded local
  file path is guaranteed to exist; it degrades gracefully to the uploader.
- **Transparency** — the "Data Quality" tab shows exactly how many rows were
  removed at each cleaning step, so nothing happens silently.
