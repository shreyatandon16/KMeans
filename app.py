
from flask import Flask, render_template, request, redirect, url_for, session, flash
import yfinance as yf
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get username and password from form
        username = request.form['username']
        password = request.form['password']

        # Check credentials
        if username == 'quantdeveloper' and password == 'devQuant@99$$':
            session['logged_in'] = True
            session.permanent = False  # Session expires when browser is closed
            return redirect(url_for('index'))
        else:
            flash('Invalid Credentials, Please try again!')  # Flash message for invalid credentials
            return redirect(url_for('login'))

    return render_template('login.html')

# Route for the main page (with authentication check)
@app.route('/')
def index():
    if not session.get('logged_in'):  # Check if the user is logged in
        return redirect(url_for('login'))

    # Define the symbol and interval
    symbol = '^NSEI'  # NIFTY 50 index symbol in Yahoo Finance
    interval = "1d"   # 1 day intervals

    # Set the date range for daily data (past year)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)

    # Fetch daily data
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval)

    # Clean the Data
    df.dropna(subset=['Close'], inplace=True)  # Drop rows where 'Close' is NaN
    df = df[df['Close'] > 0]  # Ensure no negative or zero values in 'Close'

    # Form a dataframe of the data based on the closing prices of the data
    X = np.array(df['Close'])

    # Determine the optimal number of clusters using the elbow method
    sum_of_squared_distances = []
    silhouette_scores = []
    K = range(1, 15)
    for k in K:
        km = KMeans(n_clusters=k, n_init='auto')
        km = km.fit(X.reshape(-1, 1))
        sum_of_squared_distances.append(km.inertia_)

        # Calculate silhouette score for k > 1
        if k > 1:
            c = km.predict(X.reshape(-1, 1))
            silhouette_avg = silhouette_score(X.reshape(-1, 1), c)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(None)  # No silhouette score for k=1

    # Use KneeLocator to find the optimal number of clusters
    kn = KneeLocator(K, sum_of_squared_distances, S=1.0, curve="convex", direction="decreasing")
    optimal_k = kn.knee

    # Plot the inertia values and silhouette scores with interactive and dark theme
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(K), y=sum_of_squared_distances, mode='lines+markers', name='Inertia'))
    fig.add_trace(go.Scatter(x=list(K), y=silhouette_scores, mode='lines+markers', name='Silhouette Score', yaxis='y2'))

    fig.add_vline(x=optimal_k, line=dict(color='red', dash='dash'), name='Optimal k')

    fig.update_layout(
        template='plotly_dark',
        title='Elbow Method and Silhouette Score For Optimal k',
        xaxis_title='Number of clusters (k)',
        yaxis=dict(title='Sum of squared distances', side='left'),
        yaxis2=dict(title='Silhouette Score', overlaying='y', side='right'),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),  # Adjust legend position
        margin=dict(l=50, r=50, t=50, b=50)  # Add margins to prevent overlap
    )

    graph1 = fig.to_html(full_html=False)

    # Fit the K-Means model with the optimal number of clusters if found
    if optimal_k:
        kmeans = KMeans(n_clusters=optimal_k, n_init=10).fit(X.reshape(-1, 1))
        c = kmeans.predict(X.reshape(-1, 1))

        # Calculate the silhouette score
        silhouette_avg = silhouette_score(X.reshape(-1, 1), c)

        # Find the min and max values for each cluster
        minmax = []
        for i in range(optimal_k):
            minmax.append([-np.inf, np.inf])
        for i in range(len(X)):
            cluster = c[i]
            if X[i] > minmax[cluster][0]:
                minmax[cluster][0] = X[i]
            if X[i] < minmax[cluster][1]:
                minmax[cluster][1] = X[i]

        # Visualize the clusters and their boundaries with interactive and dark theme
        colors = px.colors.qualitative.Dark24

        cluster_fig = go.Figure()

        for i in range(len(X)):
            color = colors[c[i] % len(colors)]
            cluster_fig.add_trace(go.Scatter(x=[i], y=[X[i]], mode='markers', marker=dict(color=color), showlegend=False))

        for i in range(len(minmax)):
            cluster_fig.add_hline(y=minmax[i][0], line=dict(color='green'), annotation_text=f'Resistance', annotation_position="right")
            cluster_fig.add_hline(y=minmax[i][1], line=dict(color='red'), annotation_text=f'Support', annotation_position="right")

        cluster_fig.update_layout(
            template='plotly_dark',
            title='Cluster Visualization with Support and Resistance',
            xaxis_title='Index',
            yaxis_title='Closing Price'
        )

        graph2 = cluster_fig.to_html(full_html=False)
    else:
        graph2 = "No optimal number of clusters found."

    return render_template('index.html', graph1=graph1, graph2=graph2)

# Logout route to clear the session
@app.route('/logout')
def logout():
    session.clear()  # This will clear all session data, effectively logging the user out
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

