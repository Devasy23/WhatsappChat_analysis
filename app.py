#  web app to analyse whatsapp group chats mainly and show some stats
#  it will also support topic segmentation and sentiment analysis
#  it will also have feature to show number of conversations per day and active users in that conversation
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.cluster import DBSCAN


def get_messages(cluster_number):
    return df[df['Cluster'] == cluster_number][['Sender', 'Message']]


def process_chat_data(chat_data):

    # Initialize empty lists to store the data
    dates = []
    senders = []
    messages = []

    # Iterate through each line of the chat data
    for line in chat_data:
        # Use regular expressions to extract the date and time, sender, and message
        pattern2 = r'(\d+/\d+/\d+, \d+:\d+ [ap]m) - ([^:]+): (.*)'
        pattern1 = r'(\d+/\d+/\d+, \d+:\d+ [AP]M) - ([^:]+): (.*)'

        # Test the regular expression
        match = re.match(pattern1, line.decode('utf-8'))
        match2 = re.match(pattern2, line.decode('utf-8'))
        # print(match)

        if match:
            date_time = match.group(1)
            sender = match.group(2)
            message = match.group(3)
            # print(date_time, sender, message)
            dates.append(date_time)
            senders.append(sender)
            messages.append(message)
        if match2:
            date_time = match2.group(1)
            sender = match2.group(2)
            message = match2.group(3)
            # print(date_time, sender, message)
            dates.append(date_time)
            senders.append(sender)
            messages.append(message)

    # Create a dataframe from the lists
    df = pd.DataFrame(
        {'Date & Time': dates, 'Sender': senders, 'Message': messages})
    return df


# input file
st.title("Whatsapp Chat Analyzer")
st.subheader("Upload your whatsapp chat file")
uploaded_file = st.file_uploader("Choose a file", type="txt")
if uploaded_file is not None:

    uploaded_file = uploaded_file.readlines()
    df = process_chat_data(uploaded_file)
    df['Date & Time'] = pd.to_datetime(df['Date & Time'])
    df['Date'] = df['Date & Time'].dt.date
    # time without seconds

    df['Time'] = df['Date & Time'].dt.time
    df['Time'] = df['Time'].apply(lambda x: x.hour * 60 + x.minute)
    reference_date = pd.to_datetime('1/1/2020').date()

    # Convert the 'Date' column to integers representing the number of days since the reference date
    df['Date'] = df['Date'].apply(lambda x: (reference_date - x).days)
    df['Trivial Time'] = df['Date'] * 24 * 60 + df['Time']

    X = df[['Trivial Time']]
    clustering = DBSCAN(eps=25, min_samples=25).fit(
        X, y=None, sample_weight=None)
    df['Cluster'] = clustering.labels_
    df1 = df.copy()
    # df1.drop(df1[df1['Cluster'] == -1].index, inplace = True)
    # drop columns from df1
    df1.drop(['Date & Time', 'Date', 'Time',
             'Trivial Time'], axis=1, inplace=True)
    best_clusters = df1.groupby('Cluster').count().sort_values(
        by='Message', ascending=False).reset_index().head(3)

    # calculate the top 3 most active users in best clusters
    best_clusters_active = []
    for i in range(best_clusters['Cluster'].nunique()):
        best_clusters_active.append(df1[df1['Cluster'] == best_clusters['Cluster'][i]].groupby(
            'Sender').count().sort_values(by='Message', ascending=False).reset_index())

    best_clusters_active[0].shape
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px

    # Set the style
    sns.set_style('darkgrid')

    # Create a pie chart using plotly
    fig = px.pie(df1, values='Cluster', names='Cluster',
                 title='Clusters', hole=0.5)

    # Set the figure size using the `figure_options` argument
    st.plotly_chart(fig, figure_options={'layout': {
                    'width': 5000, 'height': 5000}})

    # plt.pie(df1['Cluster'].value_counts(), labels=df1['Cluster'].value_counts().index, autopct='%1.1f%%')
    # plt.title('Clusters')

    # # Display the chart using the `st.pyplot` function
    # st.pyplot()

    # visualize the best clusters active users via pie chart
    for xyz in best_clusters_active:
        # Create a pie chart using plotly
        clusternumber = xyz['Cluster'][0]
        time = df[df['Cluster'] == clusternumber]['Date & Time'].min()
        fig = px.pie(xyz, values='Message', names='Sender',
                     title='Most Active Users in Conversation Started at ' + str(time), hole=0.5)
        if (st.button("Show Conversation " + str(clusternumber))):
            st.dataframe(get_messages(clusternumber), width=20000)
        st.plotly_chart(fig, figure_options={'layout': {
                        'width': 1000, 'height': 1000}})

    # visualize network graph of relationships between users
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.spatial.distance import euclidean
    from fastdtw import fastdtw

    def TimeSeriesDist(sender1, sender2, df):
        # Extract the time-series of activity for each sender
        sender1_activity = df[df['Sender'] == sender1]['Date & Time']
        sender2_activity = df[df['Sender'] == sender2]['Date & Time']

        # Convert the time-series of activity to a numerical representation
        reference_date = pd.to_datetime('1/1/2020').date()
        sender1_activity = (reference_date - sender1_activity.dt.date).dt.total_seconds().values
        sender2_activity = (reference_date - sender2_activity.dt.date).dt.total_seconds().values
        # # Convert the time-series of activity to numpy arrays
        sender1_activity = np.array(sender1_activity)
        sender2_activity = np.array(sender2_activity)

        sender1_activity = sender1_activity.reshape(-1, 1)
        sender2_activity = sender2_activity.reshape(-1, 1)
        # Compute the DTW distance between the time-series of activity
        distance, _ = fastdtw(sender1_activity, sender2_activity, dist=euclidean)
        
        
        return distance
    df = process_chat_data(uploaded_file)
    df["Date & Time"] = pd.to_datetime(df["Date & Time"])
    senders = df['Sender'].unique()
    # Create an empty graph
    G = nx.Graph()

    dict = {"s1": [], "s2":[], "dist": []}


    # Add the senders as nodes to the graph
    G.add_nodes_from(senders)

    # Iterate over the senders and add edges between pairs of senders
    for i, sender1 in enumerate(senders):
        for j, sender2 in enumerate(senders):
            if i == j:
                continue
            distance = TimeSeriesDist(sender1, sender2, df)
            dict["s1"].append(sender1)
            dict["s2"].append(sender2)
            dict["dist"].append(distance)
            
            G.add_edge(sender1, sender2, weight=-1*distance)

    # Set the layout and draw the graph
    # set initial positions of nodes to 0,0
    pos = nx.spring_layout(G, dim = 2, seed = 42)
    nx.draw(G, pos, with_labels=True, node_size=3000, font_size=8, font_weight='bold')
    # Display the graph
    st.pyplot()
    df12 = pd.DataFrame(dict)
    distance = dict['dist']
    df12 = df12.assign(dist_norm = 1 - (df12['dist'] - np.mean(distance))/np.std(distance))
    for i in range(len(df12)):
        G.add_edge(df12['s1'][i], df12['s2'][i], weight=df12['dist_norm'][i]*5)
        
    # Set the layout and draw the graph
    # set initial positions of nodes to 0,0
    pos = nx.spring_layout(G, dim = 2, seed = 42)
    nx.draw(G, pos, with_labels=True, node_size=3000, font_size=8, font_weight='bold')
    # Display the graph
    st.pyplot()
    # Create a plotly figure
    fig = go.Figure()
     
    from pyvis.network import Network

    # create vis network

    net = Network(notebook=True)
    # load the networkx graph

    net.from_nx(G)
    # show
    net.show("example.html")
    # Read the HTML code from the file
    with open('example.html', 'r') as f:
        html_code = f.read()

    # Display the HTML code in the Streamlit app
    st.markdown(html_code, unsafe_allow_html=True)


