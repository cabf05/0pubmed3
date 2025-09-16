import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="Medical Hot Topics Explorer", layout="wide")

st.title("📊 Medical Hot Topics Explorer")
st.markdown("This tool fetches PubMed articles, ranks them, and detects **emerging hot topics** per medical area.")

# -------------------- Inputs --------------------
st.sidebar.header("Search Configuration")

# Área médica -> define query básica
area = st.sidebar.selectbox(
    "Select Medical Area",
    ["Endocrinology", "Cardiology", "Oncology"]
)

query_map = {
    "Endocrinology": '("Endocrinology" OR "Diabetes") AND 2024/01/01:2025/09/01[Date - Publication]',
    "Cardiology": '("Cardiology" OR "Myocardial Infarction") AND 2024/01/01:2025/09/01[Date - Publication]',
    "Oncology": '("Oncology" OR "Cancer") AND 2024/01/01:2025/09/01[Date - Publication]'
}
query = query_map[area]

max_results = st.sidebar.number_input("Max number of articles", min_value=50, max_value=1000, value=300, step=50)

# -------------------- Run search --------------------
if st.sidebar.button("🔎 Run PubMed Search"):
    with st.spinner("Fetching articles..."):

        # Step 1: Use esearch to get PMIDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "retmax": str(max_results),
            "retmode": "json",
            "term": query
        }
        r = requests.get(search_url, params=search_params)
        id_list = r.json()["esearchresult"].get("idlist", [])

        if not id_list:
            st.warning("No articles found for this query.")
            st.stop()

        # Step 2: Use efetch in batch
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml"
        }
        response = requests.get(efetch_url, params=params, timeout=30)

        records = []

        try:
            root = ET.fromstring(response.content)
            articles = root.findall(".//PubmedArticle")
            for article in articles:
                try:
                    pmid = article.findtext(".//PMID")
                    title = article.findtext(".//ArticleTitle", "")
                    abstract = " ".join([abst.text for abst in article.findall(".//AbstractText") if abst is not None])
                    link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    journal = article.findtext(".//Journal/Title", "")
                    year = article.findtext(".//PubDate/Year") or "N/A"

                    records.append({
                        "PMID": pmid,
                        "Title": title,
                        "Abstract": abstract,
                        "Link": link,
                        "Journal": journal,
                        "Year": year
                    })
                except Exception:
                    pass
        except Exception:
            st.error("Failed to parse XML from PubMed.")

        df = pd.DataFrame(records)

        st.success(f"Fetched {len(df)} articles for {area}.")

        if not df.empty:
            # -------------------- PART 1: Keyword Frequency --------------------
            st.header("📌 Keyword Frequency Analysis")

            text_data = " ".join(df["Title"].astype(str).tolist() + df["Abstract"].astype(str).tolist())
            text_data = text_data.lower()
            words = re.findall(r"\b[a-z]{3,}\b", text_data)

            stop_words = set(ENGLISH_STOP_WORDS)
            words = [w for w in words if w not in stop_words]

            freq = Counter(words)
            top_terms = freq.most_common(30)

            freq_df = pd.DataFrame(top_terms, columns=["Term", "Frequency"])
            st.dataframe(freq_df, use_container_width=True)

            # Wordcloud
            st.subheader("Word Cloud of Most Frequent Terms")
            wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(freq)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # -------------------- PART 2: Temporal Trends --------------------
            st.header("📈 Temporal Trends of Top Terms")

            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            df = df.dropna(subset=["Year"])

            trend_terms = [t[0] for t in top_terms[:5]]  # top 5 terms
            trend_data = {term: [] for term in trend_terms}
            years = sorted(df["Year"].unique())

            for year in years:
                year_text = " ".join(
                    df[df["Year"] == year]["Title"].astype(str).tolist() +
                    df[df["Year"] == year]["Abstract"].astype(str).tolist()
                ).lower()
                for term in trend_terms:
                    trend_data[term].append(year_text.count(term))

            trend_df = pd.DataFrame(trend_data, index=years)
            st.line_chart(trend_df)

            # -------------------- PART 3: Show Articles --------------------
            st.header("📰 Articles Retrieved")
            st.dataframe(df[["Title", "Journal", "Year", "Link"]], use_container_width=True)
