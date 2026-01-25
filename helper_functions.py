import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
import io

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import streamlit as st

import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib, json
import numpy as np

def load_data():
    return pd.read_csv('data.csv')



def load_data_GD_API(file_id):

    """
    Loads data from Google Drive using the service account stored in Streamlit secrets.
    """
    
    # Define the required scopes (modify if needed)
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    # Authenticate using service account credentials
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)

    # Build the Google Drive API service
    service = build("drive", "v3", credentials=creds)

    request = service.files().export_media(fileId=file_id, mimeType='text/csv')
    file_stream = io.BytesIO(request.execute())

    df = pd.read_csv(file_stream)

    return df

    
def preprocess_data(raw_data, numeric_columns=None, columns_to_drop=None):
    """
    Preprocess the input raw_data DataFrame with specified transformations.

    Parameters:
        raw_data (pd.DataFrame): The raw input DataFrame to preprocess.
        emoji_to_int (dict): Dictionary mapping emoji strings to integers.
        columns_to_drop (list): List of columns to drop from the DataFrame (optional).

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for modeling.
    """
    data = raw_data.copy()

    # Convert specified columns to numeric, if provided
    if numeric_columns:
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')  # Coerce invalid values to NaN

    # Create dictionaries for values that are 'emojis'

    emoji_to_int_mood = {':(':0, ':/':1, ':)':2}
    emoji_to_int_gym = {'No Gym':0, ':(':1, ':/':2, ':)':3}

    # Map emojis to integer values
    data['Mood of the Day'] = data['Mood of the Day'].map(emoji_to_int_mood)
    data['Gym Motivation'] = data['Gym Motivation'].map(emoji_to_int_gym)
    data['Gym'] = (data['Gym Motivation']>0).astype(int)

    # Drop unnecessary columns
    if columns_to_drop:
        data.drop(columns_to_drop, axis=1, inplace=True)

    # Convert 'Location' to a categorical type
    data['Location'] = data['Location'].astype('category')

    # Ensure the Date column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Sort the DataFrame by the Date column
    data = data.sort_values(by='Date')

    # Add a column for the day of the week
    data['Day of Week'] = data['Date'].dt.day_name()

    # Reset the index
    data = data.reset_index(drop=True)

    # Create lagged columns for previous day values
    data['Healthy Eats Previous Day'] = data['Healthy Eats'].shift(1)
    data['Alcohol Previous Day'] = data['Alcohol?'].shift(1)
    data['Weed Previous Day'] = data['Bread?'].shift(1)
    data['Gym Previous Day'] = data['Gym'].shift(1)
    data['Mood Previous Day'] = data['Mood of the Day'].shift(1)

    # Calculate rolling sums for specific columns
    data['Vitamins Last 7 Days'] = data['Vitamins?'].shift(1).rolling(window=7).sum().fillna(0)
    data['Hours of Sleep Last 4 Days'] = data['Hours of Sleep'].rolling(window=4).sum()
    data['Creatine Last 7 Days'] = data['Creatine?'].shift(1).rolling(window=7).sum()

    # Transform 'Mood of the Day' to binary
    data['Mood of the Day Binary'] = data['Mood of the Day'].apply(lambda x: 1 if x == 2 else 0)
    data['Mood Previous Day Binary'] = data['Mood Previous Day'].apply(lambda x: 1 if x == 2 else 0)

    # Drop rows with missing values
    data = data.dropna()

    return data

def barchart_pred_vs_target(pred_column_name, target_column_name, df):
    """
    Create a bar chart for the mean of a target variable grouped by a predictor,
    ensuring proper handling of categorical predictors,
    and include Wilson score confidence intervals for proportions.

    Parameters:
        pred_column_name (str): The column name of the predictor variable.
        target_column_name (str): The column name of the target variable (binary).
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    df = df.copy()

    # Ensure proper day order if grouping by Day of Week
    if pred_column_name == 'Day of Week':
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df[pred_column_name] = pd.Categorical(df[pred_column_name], categories=days_order, ordered=True)

    # Convert the predictor column to categorical (ensures proper x-axis spacing)
    df[pred_column_name] = pd.Categorical(df[pred_column_name], ordered=True)

    # Group by the predictor and calculate the count of successes and total observations
    grouped = df.groupby(pred_column_name)[target_column_name].agg(['sum', 'count']).reset_index()
    grouped = grouped[grouped['count'] >= 1]
    grouped['mean'] = grouped['sum'] / grouped['count']  # Calculate the mean (proportion)

    # Calculate confidence intervals
    grouped['std'] = df.groupby(pred_column_name)[target_column_name].std().reset_index(drop=True)
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['ci_lower'] = grouped['mean'] - 1.96 * grouped['se']
    grouped['ci_upper'] = grouped['mean'] + 1.96 * grouped['se']

    # Visualization: Bar chart with rounded bars and proper x-axis
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("Set2", len(grouped))  

    # Create bars with black outline
    bars = ax.bar(
        grouped[pred_column_name].astype(str),  # Convert categories to strings for proper labeling
        grouped['mean'],
        color=colors,  # Color each bar
        edgecolor="black",  # Black outline
        linewidth=1.5,  # Outline thickness
        width=0.7,  # Width of the bars
        zorder=2  # Ensure bars are on top of the grid
    )

    # Add error bars
    ax.errorbar(
        x=range(len(grouped)),  # X-axis positions
        y=grouped['mean'],  # Proportions
        yerr=[
            grouped['mean'] - grouped['ci_lower'],  # Lower error
            grouped['ci_upper'] - grouped['mean'],  # Upper error
        ],
        fmt='none',  # No line connecting the points
        ecolor='black',  # Error bar color
        elinewidth=1,  # Thickness of error lines
        capsize=4,  # Size of caps
        zorder=3  # Draw error bars on top of bars
    )

    # Add annotations for bar values
    for bar, value, color in zip(bars, grouped['mean'], colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Center of the bar
            bar.get_height(), 
            f'{value:.2f}',  # Format the value
            ha='center',  # Center-aligned
            fontsize=12,
            fontweight='bold',
            color=color,
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'),  # White box
        )

    # Beautify the chart
    ax.set_xlabel(pred_column_name, fontsize=14)
    ax.set_ylabel(f'Proportion of {target_column_name}', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    fig.tight_layout()
    plt.close(fig)  # Optional but avoids memory leaks in Streamlit
    return fig




def line_chart(df, column_name, column):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    avg_last_30 = df[column_name].iloc[-37:-7].mean()
    avg_last_7 = df[column_name].iloc[-7:].mean()

    fig = go.Figure()

    # Line chart for the last 7 days
    fig.add_trace(go.Scatter(
        x=df['Date'][-7:], 
        y=df[column_name][-7:], 
        mode='lines+markers',
        marker=dict(size=12, color='royalblue', line=dict(color='white', width=1.5)),
        line=dict(color='royalblue', width=2),
        name='Last 7 Days',
        hovertemplate=(
            "<b></b> %{y:.1f}<br>"
        )
    ))

    # Horizontal line for 7-day average
    fig.add_trace(go.Scatter(
        x=df['Date'][-7:], 
        y=[avg_last_7] * 7,
        mode='lines',
        line=dict(color='royalblue', width=2, dash='dash'),
        name='7-Day Avg',
        hovertemplate=(
            "<b></b> %{y:.1f}<br>"
        )
    ))

    # Horizontal line for 30-day average
    fig.add_trace(go.Scatter(
        x=df['Date'][-7:], 
        y=[avg_last_30] * 7,
        mode='lines',
        line=dict(color='darkorange', width=2, dash='dash'),
        name='30-Day Avg',
        hovertemplate=(
            "<b></b> %{y:.1f}<br>"
        )
    ))

    # Layout adjustments for better visualization
    fig.update_layout(
        title=dict(text=column_name, font=dict(size=16, family="Arial")),
        xaxis=dict(title='Date', tickformat="%b %d"),
        yaxis=dict(title=column_name),
        hovermode="x",  # Combine all tooltips for the same x value
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),  # Add some spacing
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            align="left"
        )
    )

    # Display the chart in the provided column
    column.plotly_chart(fig)


def calculate_yesterday_val(df, column_name):
    """
    description...
    """
    df = df.copy()

    yesterday_val = df[column_name].iloc[-1]

    week_avg = df[column_name].iloc[-8:-1].mean()

    percentage_change = int(((yesterday_val - week_avg)/week_avg) * 100)

    return (yesterday_val, percentage_change)


def create_metric_st(yesterday_val, percentage_change, title, container):

    # Determine Color
    color = "green" if percentage_change > 0 else "orange"
    high_low = "Higher" if percentage_change > 0 else "Lower"

    # Streamlit UI inside the container
    with container:
        st.markdown(f"""
            <div style='display: flex; flex-direction: column; align-items: center; text-align: center;'>
                <h3 style='margin: 0;'>{title}</h3>
                <h1 style='font-size: 60px; margin: 0 auto;'>{yesterday_val}</h1>
                <p style='font-size: 16px; color: {color}; font-weight: normal; margin: 0; display: flex; flex-direction: column; align-items: center;'>
                    <span style='font-weight: bold; font-size: 18px;'>{percentage_change:.0f}%\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0</span>
                    {high_low} than last week's avg.\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0
                </p>
            </div>
        """, unsafe_allow_html=True)

def preprocess_for_logreg_manual(raw_data, numeric_columns=None, columns_to_drop=None):
    """
    Preprocess data to match the handpicked multinomial logistic regression features.
    Returns the engineered feature frame (no target column).
    """
    data = raw_data.copy()

    if columns_to_drop:
        data.drop(columns_to_drop, axis=1, inplace=True)

    if numeric_columns:
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date').reset_index(drop=True)

    data['Day_sin'] = np.sin(2 * np.pi * data['Date'].dt.weekday / 7)
    data['Day_cos'] = np.cos(2 * np.pi * data['Date'].dt.weekday / 7)
    data['Is Weekend'] = data['Date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int)

    emoji_to_int_mood = {':(':0, ':/':1, ':)':2}
    emoji_to_int_gym = {'No Gym':0, ':(':1, ':/':2, ':)':3}
    data['Mood of the Day'] = data['Mood of the Day'].map(emoji_to_int_mood)
    data['Gym Motivation'] = data['Gym Motivation'].map(emoji_to_int_gym)
    data['Gym'] = (data['Gym Motivation'] > 0).astype(int)

    data['Vitamins Rolling Sum 7'] = data['Vitamins?'].rolling(window=7).sum().fillna(0)
    data['Creatine Rolling Sum 7'] = data['Creatine?'].rolling(window=7).sum()

    data['Sleep EWM'] = data['Hours of Sleep'].ewm(span=5).mean()
    data['Mood EWM'] = data['Mood of the Day'].ewm(span=5).mean()
    data['Healthy Eats EWM'] = data['Healthy Eats'].ewm(span=5).mean()
    data['Gym Motivation EWM'] = data['Gym Motivation'].ewm(span=5).mean()

    data['Caffeine Consumption'] = data['Caffeine Consumption'].fillna(0)

    data = data.dropna()
    return data

def load_logreg_manual_model(model_path="models/logreg_manual_full.pkl", feature_path="models/logreg_manual_features.json"):
    model = joblib.load(model_path)
    with open(feature_path) as f:
        features = json.load(f)
    return model, features

def predict_good_mood_probability(logreg_model, features, feature_frame):
    """
    Returns per-class probabilities and the 'good mood' probability (class=2) for the latest row.
    """
    row = feature_frame[features].iloc[[-1]]
    probs = logreg_model.predict_proba(row)[0]
    return probs, probs[2]


def cumulative_last_30_days_comparison(df, column_to_sum):
    """
    Generates a cumulative Plotly line chart comparing the last 30 days vs the previous 30 days.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'Date' column in datetime format.
        column_to_sum (str): Column to aggregate cumulatively.

    Returns:
        fig (plotly.graph_objs._figure.Figure): Plotly figure with the plot.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    today = pd.Timestamp.now().normalize()
    current_30_start = today - pd.Timedelta(days=29)
    previous_30_start = current_30_start - pd.Timedelta(days=30)
    previous_30_end = current_30_start - pd.Timedelta(days=1)

    # Filter data
    current_30_df = df[(df['Date'] >= current_30_start) & (df['Date'] <= today)].copy()
    previous_30_df = df[(df['Date'] >= previous_30_start) & (df['Date'] <= previous_30_end)].copy()

    # Assign day indices (0 to 29)
    current_30_df['DayIndex'] = (current_30_df['Date'] - current_30_start).dt.days
    previous_30_df['DayIndex'] = (previous_30_df['Date'] - previous_30_start).dt.days

    # Group and compute cumulative sums
    current_cum = current_30_df.groupby('DayIndex')[column_to_sum].sum().sort_index().cumsum()
    previous_cum = previous_30_df.groupby('DayIndex')[column_to_sum].sum().sort_index().cumsum()

    # Reindex to align both series
    days_range = range(30)
    current_cum = current_cum.reindex(days_range, fill_value=np.nan)
    previous_cum = previous_cum.reindex(days_range, fill_value=np.nan)

    previous_dates = [previous_30_start + pd.Timedelta(days=i) for i in range(30)]
    current_dates = [current_30_start + pd.Timedelta(days=i) for i in range(30)]

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(30)),
        y=previous_cum,
        mode='lines',
        name='Previous 30 Days',
        line=dict(color='gray', dash='dash'),
        marker=dict(size=8),
        customdata=np.array(previous_dates),  # attaches dates to each point
        hovertemplate="Date %{customdata|%b %d}: %{y:.1f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=list(range(30)),
        y=current_cum,
        mode='lines',
        name='Last 30 Days',
        line=dict(color='royalblue'),
        marker=dict(size=8),
        customdata=np.array(current_dates),
        hovertemplate="Date %{customdata|%b %d}: %{y:.1f}<extra></extra>"
    ))

    fig.update_layout(
        xaxis_title='Day (0 = Earliest)',
        yaxis_title=f'Cumulative {column_to_sum}',
        hovermode='closest',
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        )
    )

    return fig



# --- AI helper: build prompt + call OpenAI ---

from openai import OpenAI
import os

# ---- secrets loader ----
def _load_openai_key():
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        return st.secrets["openai"]["api_key"]
    if "openai_api_key" in st.secrets:
        return st.secrets["openai_api_key"]
    return os.getenv("OPENAI_API_KEY")

# ---- small utils ----
def classify_mood_label(prob: float) -> str:
    if prob >= 0.85:
        return "good"
    elif prob >= 0.75:
        return "ok"
    elif prob >= 0.50:
        return "not so good"
    else:
        return "bad"

def last30_averages_from_x(x: pd.DataFrame) -> dict:
    """Compute 30-day (last 30 rows) averages directly from x."""
    last30 = x.tail(30).copy()
    return {
        "sleep": float(last30["Hours of Sleep"].mean()),
        "gym_rate": float(last30["Gym"].mean()),
        "healthy_rate": float(last30["Healthy Eats"].mean()),
        "mood_avg": float(last30["Mood of the Day"].mean()),
    }

def logreg_good_class_contributions(model, features: list, row_df: pd.DataFrame) -> pd.Series:
    """
    For multinomial logistic regression, compute class-2 (good mood) logit contributions:
    coefficient * feature_value for the provided row.
    Returns a Series sorted by absolute contribution descending.
    """
    row = row_df[features].iloc[0]
    coef = pd.Series(model.coef_[2], index=features)  # class 2 row of coef matrix
    contrib = coef * row
    return contrib.reindex(contrib.abs().sort_values(ascending=False).index)


# ---- prompt builder using x for averages and new_data for yesterday ----
def build_mood_prompt_from_x(
    x: pd.DataFrame,
    new_data: pd.DataFrame | pd.Series,
    predicted_probability: float,
    drivers: pd.Series,
    averages: dict | None = None,
) -> str:
    """
    Build the LLM prompt using:
      - x: full feature frame (for 30-day averages)
      - new_data: the already-built single-row input for 'tomorrow' (yesterday's current values)
      - predicted_probability: precomputed model probability for 'good' mood
      - drivers: coefficient*value contributions Series (index = feature names)
      - averages: optional dict; if None, computed from x
    """
    # get the single row as a Series
    if isinstance(new_data, pd.DataFrame):
        y = new_data.iloc[0]
    else:
        y = new_data

    avg = averages or last30_averages_from_x(x)
    label = classify_mood_label(predicted_probability)
    driver_lines = "\n  • ".join([f"{feat}: {val:+.3f}" for feat, val in drivers.items()])

    prompt = f"""
Context:
- Yesterday’s stats:
  • Hours of Sleep: {float(y['Hours of Sleep']):.2f}
  • Gym (1=yes,0=no): {int(y['Gym'])}
  • Healthy Eats (self-rated, 1–10): {int(y['Healthy Eats'])}
  • Mood of the Day (2=good,1=neutral,0=bad): {int(y['Mood of the Day'])}

- Last 30 days averages (from tracked metrics):
  • Avg Hours of Sleep: {avg['sleep']:.2f}
  • Gym rate: {avg['gym_rate']:.2f}
  • Healthy Eats rate: {avg['healthy_rate']:.2f}
  • Avg Mood: {avg['mood_avg']:.2f}

- Model output for tomorrow:
  • Probability mood=good: {predicted_probability:.3f}, consider this {label}

- Feature contributions from the logistic regression (feature → coef*value for class=good):
  • {driver_lines}

Task:
- Summary: State tomorrow’s predicted mood and probability and explain briefly why, referencing the feature contributions and how current values compare to the 30-day averages (paraphrase as needed; no new variables).
- Recommendations: Based only on the measured variables (Hours of Sleep, Gym, Healthy Eats), explain: 
One: What I’m currently doing well that supports a good mood. 
Two: What I could adjust in these same variables to boost my mood the day after tomorrow.

Note:
Just title of segment ('Summary' or 'Recommendations') and your response. Do not add numbers.
Make recommendations specific and measurable, tied directly to the data provided.
Keep your answers brief, 2-4 sentences each.
"""
    return prompt

# ---- system prompt + caller ----
SYSTEM_PROMPT = """
You are a professional medical expert with decades of experience in psychiatry, psychology, and wellness science.
You combine deep scientific and clinical knowledge with advanced expertise in data science and machine learning.
Your role is to interpret mood predictions, explain probabilities and mood drivers and insights clearly, discuss likely causal effects
of variables (sleep, exercise, nutrition), and provide practical, evidence-informed recommendations. Be rigorous, concise,
and data-driven. Do not make unsupported medical claims or diagnoses; limit suggestions to safe lifestyle adjustments.
You are a top notch scientist: confident and data oriented.
"""

def call_ai_mood_explainer(prompt: str, model_name: str = "gpt-4o-mini") -> str:
    key = _load_openai_key()
    if not key:
        raise RuntimeError("OpenAI API key not found in st.secrets or OPENAI_API_KEY.")
    import os

    # Prevent Streamlit's proxy injection issue
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("ALL_PROXY", None)

    from httpx import Client as HTTPClient
    client = OpenAI(api_key=key, http_client=HTTPClient(trust_env=False))

    resp = client.chat.completions.create(  # if your SDK uses .chat.completions.create, keep that
        model=model_name,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    # if you're on the new SDK, adjust the accessor accordingly
    return resp.choices[0].message.content
