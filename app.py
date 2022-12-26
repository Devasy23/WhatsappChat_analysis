#  web app to analyse whatsapp group chats mainly and show some stats
#  it will also support topic segmentation and sentiment analysis
#  it will also have feature to show number of conversations per day and active users in that conversation
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  function to read the file
def read_file(file):
    df = pd.read_csv(file)
    return df

#  function to show the data
def show_data(df):
    if st.checkbox("Show Data"):
        st.subheader("Data")
        st.write(df)
        
#  display input for file upload
st.title("Whatsapp Chat Analysis")
st.subheader("Upload your chat file")
file = st.file_uploader("Upload your file", type=["csv"])
if file is not None:
    df = read_file(file)
    show_data(df)
    st.success("File Uploaded Successfully")
else:
    st.warning("Please upload a file")
     
#  function to show the number of messages per day
def show_messages_per_day(df):
    if st.checkbox("Show Messages per Day"):
        st.subheader("Messages per Day")
        df["Date"] = df["Date"].apply(lambda x: x.split(",")[0])
        df["Date"] = pd.to_datetime(df["Date"])
        df["Date"] = df["Date"].dt.date
        df = df.groupby("Date").count()
        df = df.reset_index()
        df = df[["Date", "Message"]]
        df = df.rename(columns={"Message": "Messages"})
        st.line_chart(df)
        
show_messages_per_day(df)
