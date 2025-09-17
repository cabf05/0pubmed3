import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

# -------------------- Streamlit Config --------------------
st.set_page_config(page_title="Medical Hot Topics Explorer", layout="wide")
st.title("🔍 Medical Hot Topics Explorer")
st.markdown("This tool fetches PubMed articles, ranks them, and detects emerging hot topics per medical area.")

# -------------------- Inputs --------------------
st.header("Step 1: Customize the Search")

default_query = '("Endocrinology" OR "Diabetes") AND 2024/10/01:2025/09/01[Date - Publication]'
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

max_results = st.number_input("Max number of articles to fetch", min_value=10, max_value=500, value=200, step=10)

hot_keywords = ["glp-1", "semaglutide", "tirzepatide", "ai", "machine learning", "telemedicine"]

# -------------------- PubMed Fetch --------------------
if st.button("🔎 Run Analysis"):
    with st.spinner("Fetching articles from PubMed..."):

        # Step 1: Get PMIDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "retmax": str(max_results),
            "retmode": "json",
            "term": query
        }
        r = requests.get(search_url, params=search_params)
        id_list = r.json()["esearchresult"].get("idlist", [])

        # Step 2: Fetch articles
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml"
        }
        response = requests.get(efetch_url, params=params, timeout=30)

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
            st.subheader("Ranked Articles")
            st.dataframe(df[["Title", "Journal", "Date", "Score", "Why"]], use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download CSV", data=csv, file_name="ranked_pubmed_results.csv", mime="text/csv")
        else:
            st.warning("No valid articles found to display.")

        # -------------------- Word Cloud --------------------
        st.header("☁️ Word Cloud of Titles")

        text_data = " ".join(df["Title"].astype(str).tolist())
        stop_words = set(ENGLISH_STOP_WORDS)
        wordcloud = WordCloud(width=800, height=400, stopwords=stop_words, background_color="white").generate(text_data)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # -------------------- Frequency Analysis --------------------
        st.header("📊 Keyword Frequency")
        vectorizer = CountVectorizer(stop_words='english', max_features=30)
        X = vectorizer.fit_transform(df["Title"].astype(str).tolist())
        freqs = zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0))
        freq_df = pd.DataFrame(freqs, columns=["Keyword", "Frequency"]).sort_values("Frequency", ascending=False)

        st.dataframe(freq_df, use_container_width=True)
        fig2, ax2 = plt.subplots()
        ax2.bar(freq_df["Keyword"], freq_df["Frequency"])
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)

        # -------------------- Advanced Hot Topics (SciSpacy) --------------------
        st.header("🧬 Advanced Hot Topics Analysis")

        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")

            doc = nlp(text_data)
            entities = [ent.text.lower() for ent in doc.ents]

            # Generic terms to filter
            generic_med_terms = set([
                "study", "patient", "patients", "group", "results", "trial", "clinical",
                "analysis", "effect", "treatment", "observed", "report", "case", "cohort"
            ])

            entity_text = " ".join(entities)
            vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3), max_features=100)
            X = vectorizer.fit_transform([entity_text])
            terms = vectorizer.get_feature_names_out()
            freq = X.toarray().sum(axis=0)

            filtered_terms = [(term, count) for term, count in zip(terms, freq) if term not in generic_med_terms]
            top_terms = sorted(filtered_terms, key=lambda x: x[1], reverse=True)[:30]

            st.subheader("Top 30 Biomedical Terms (NER + Ngrams)")
            st.dataframe(pd.DataFrame(top_terms, columns=["Term", "Frequency"]))

            if top_terms:
                term_freq_dict = dict(top_terms)
                wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(term_freq_dict)
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                ax3.imshow(wc, interpolation="bilinear")
                ax3.axis("off")
                st.pyplot(fig3)

        except Exception as e:
            st.warning(f"Could not run advanced analysis: {e}")
