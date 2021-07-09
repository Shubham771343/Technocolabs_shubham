from google.protobuf import message
import streamlit as st
import spacy
from textblob import TextBlob


def main():
    st.title("Sentiment analysis of Amazon product review")
    st.subheader("Amazon_comment_sentimet_Analysis")

    # sentiment Analysis
    if st.checkbox("Show Sentiment Analysis"):
        st.subheader("sentiment of your comment")
        message = st.text_area("Enter your text","Type Here")
        if st.button("Annalyze"):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)





if __name__ == '__main__':
    main()
