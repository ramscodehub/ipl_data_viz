# 🏏 IPL Cricket Stats Visualization

An interactive dashboard to explore and analyze IPL (Indian Premier League) cricket statistics using **Python**, **Dash**, and **Plotly**. This project includes dynamic visualizations for player performance trends and real-time comparisons.

🔗 **Live App**: [IPL Dashboard on Google Cloud Run](https://dashapp-lxnqkebi7q-ue.a.run.app/)

## 🚀 Features

- 📊 **Interactive Visualizations**: Batting and bowling stats over seasons, player performance trends, and head-to-head analysis.
- 🧩 **User-friendly Interface**: Built with Dash and Plotly for smooth navigation and interactivity.
- 🐳 **Dockerized**: Easily deployable using Docker and Google Cloud.

## 🛠️ Tech Stack

- **Python 3.8**
- **Dash + Plotly**
- **Pandas**
- **Google Cloud Platform**
- **Docker**

## 📦 Deployment

To run locally using Docker:

```bash
# Build Docker image
docker build -t ipl-dashboard .

# Run the Docker container
docker run -p 8050:8050 ipl-dashboard
```

## 📝 Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
