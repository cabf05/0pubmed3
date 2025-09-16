import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from collections import Counter
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="PubMed Relevance Ranker", layout="wide")

st.title("🔍 PubMed Relevance Ranker")
st.markdown("This tool fetches articles from PubMed and ranks them based on their **potential relevance**, using only metadata from the PubMed XML.")

# -------------------- Inputs --------------------
st.header("Step 1: Customize the Search")

default_query = '("Endocrinology" OR "Diabetes") AND 2024/10/01:2025/06/28[Date - Publication]'
query = st.text_area("PubMed Search Query", value=default_query, height=100)

default_journals = "\n".join([
    "N Engl J Med", "JAMA", "BMJ", "Lancet", "Nature", "Science", "Cell"
])
journal_input = st.text_area("High-Impact Journals (one per line)", value=default_journals, height=150)
journals = [j.strip().lower() for j in journal_input.strip().split("\n") if j.strip()]

default_institutions = "\n".join([
    "Harvard", "Oxford", "Mayo Clinic", "NIH", "Stanford",
    "UCSF", "Yale", "Cambridge", "Karolinska", "Johns Hopkins"
])
inst_input = st.text_area("Renowned Institutions (one per line)", value=default_institutions, height=150)
institutions = [i.strip().lower() for i in inst_input.strip().split("\n") if i.strip()]

max_results = st.number_input("Max number of articles to fetch", min_value=10, max_value=1000, value=250, step=10)

hot_keywords = ["glp-1", "semaglutide", "tirzepatide", "ai", "machine learning", "telemedicine"]

# -------------------- Run PubMed Search --------------------
if st.button("🔎 Run PubMed Search"):
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
        response = requests.get(efetch_url, params=params, timeout=20)

        parsed_ok = 0
        parsed_fail = 0
        records = []

        def score_article(article):
            score = 0
            reasons = []

            journal = article.findtext(".//Journal/Title", "").lower()
            if any(j in journal for j in journals):
                score += 2
                reasons.append("High-impact journal (+2)")

            pub_types = [pt.text.lower() for pt in article.findall(".//PublicationType")]
            valued_types = ["randomized controlled trial", "systematic review", "meta-analysis", "guideline", "practice guideline"]
            if any(pt in valued_types for pt in pub_types):
                score += 2
                reasons.append("Valued publication type (+2)")

            authors = article.findall(".//Author")
            if len(authors) >= 5:
                score += 1
                reasons.append("Multiple authors (+1)")

            affiliations = [aff.text.lower() for aff in article.findall(".//AffiliationInfo/Affiliation") if aff is not None]
            if any(inst in aff for aff in affiliations for inst in institutions):
                score += 1
                reasons.append("Prestigious institution (+1)")

            title = article.findtext(".//ArticleTitle", "").lower()
            if any(kw in title for kw in hot_keywords):
                score += 2
                reasons.append("Hot keyword in title (+2)")

            if article.find(".//GrantList") is not None:
                score += 2
                reasons.append("Has research funding (+2)")

            return score, "; ".join(reasons)

        try:
            root = ET.fromstring(response.content)
            articles = root.findall(".//PubmedArticle")
            for article in articles:
                try:
                    pmid = article.findtext(".//PMID")
                    title = article.findtext(".//ArticleTitle", "")
                    link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    journal = article.findtext(".//Journal/Title", "")
                    date = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate") or "N/A"
                    score, reason = score_article(article)
                    records.append({
                        "Title": title,
                        "Link": link,
                        "Journal": journal,
                        "Date": date,
                        "Score": score,
                        "Why": reason
                    })
                    parsed_ok += 1
                except Exception:
                    parsed_fail += 1
        except Exception:
            st.error("Failed to parse XML from PubMed.")

        df = pd.DataFrame(records).sort_values("Score", ascending=False)

        st.success(f"Found {len(id_list)} PMIDs. Successfully parsed {parsed_ok} articles. Failed to parse {parsed_fail}.")

        if not df.empty:
            # -------------------- Show Articles --------------------
            st.dataframe(df[["Title", "Journal", "Date", "Score", "Why"]], use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download CSV", data=csv, file_name="ranked_pubmed_results.csv", mime="text/csv")

            # -------------------- Hot Topics Analysis --------------------
            st.header("🔥 Hot Topics Analysis")

            text_data = " ".join(df["Title"].astype(str).tolist()).lower()
            words = re.findall(r"\b[a-z]{3,}\b", text_data)
            stop_words = set(ENGLISH_STOP_WORDS)
            words = [w for w in words if w not in stop_words]

            freq = Counter(words)
            top_terms = freq.most_common(30)

            st.subheader("Top 30 Frequent Terms")
            st.dataframe(pd.DataFrame(top_terms, columns=["Term", "Frequency"]))

            wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(freq)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
