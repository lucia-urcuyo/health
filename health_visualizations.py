import helper_functions as hf
import streamlit as st
import pandas as pd


file_id = '1E8QYsgwoIFwLbG4hNOiqGWzHK4yy8uOHm6Yg7nBXodQ'

raw_data = hf.load_data_GD_API(file_id)
print(raw_data)

# raw_data = hf.load_data()
# raw_data = raw_data.drop(columns=["Unnamed: 0"])


# Columns that should be numeric
numeric_columns = ['Caffeine Consumption', 'Vitamins?', 'Creatine?', 'Alcohol?', 'Bread?', 'Hours of Sleep', 'Healthy Eats']

#Columns to drop
columns_to_drop = ['Timestamp']

data = hf.preprocess_data(raw_data, numeric_columns, columns_to_drop)

# App Title
st.set_page_config(page_title="My Dashboard", layout="centered", initial_sidebar_state="collapsed")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Monthly Comparison", "Mood Drivers and Insights"])



# Define Each Page
def home():
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 1600px;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 style='text-align: center;'>Mood Prediction Dashboard</h1>", unsafe_allow_html=True)

    # Prepare data for prediction
    # Preprocess for multinomial logistic regression (handpicked features)
    feature_frame = hf.preprocess_for_logreg_manual(raw_data, numeric_columns, columns_to_drop)
    logreg_model, features = hf.load_logreg_manual_model()
    probs_all, predicted_probability = hf.predict_good_mood_probability(logreg_model, features, feature_frame)
    new_data = feature_frame.iloc[[-1]]

    # Determine color based on predicted value
    gradient_color = f"rgb({255 - int(predicted_probability * 255)}, {int(predicted_probability * 255)}, 0)"

     # Display prediction prominently at the top with custom styling
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <h2 style="font-size: 32px; margin-bottom: 0; font-family: Arial, sans-serif; color: black;">Tomorrow's Predicted Good Mood Probability</h2>
            <p style="font-size: 60px; margin-top: 0; font-family: Arial, sans-serif; font-weight: bold; color: {gradient_color};">
                {predicted_probability:.0%}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- AI Mood Explainer ---
    drivers_series = hf.logreg_good_class_contributions(logreg_model, features, new_data)
    avg_dict = hf.last30_averages_from_x(feature_frame)

    prompt = hf.build_mood_prompt_from_x(
        x=feature_frame,
        new_data=new_data,
        predicted_probability=predicted_probability,
        drivers=drivers_series,
        averages=avg_dict,
    )

    if st.button("Explain tomorrow’s mood & suggest 3 actions"):
        with st.spinner("Thinking..."):
            text = hf.call_ai_mood_explainer(prompt)
        st.markdown(text)

    # A Look to Yesterday's Stats Section
    st.header("A Look to Yesterday's Stats")

    # Create three columns
    col1, col2, col3 = st.columns(3)

    yesterday_val, percentage_change = hf.calculate_yesterday_val(data, 'Hours of Sleep')
    hf.create_metric_st(yesterday_val, percentage_change, "Hours of Sleep", col1)

    yesterday_val, percentage_change = hf.calculate_yesterday_val(data, 'Gym')
    hf.create_metric_st(yesterday_val, percentage_change, "Gym", col2)

    yesterday_val, percentage_change = hf.calculate_yesterday_val(data, 'Healthy Eats')
    hf.create_metric_st(yesterday_val, percentage_change, "Healthy Eats", col3)

    st.header("Trends")


    hf.line_chart(data, 'Hours of Sleep', st)
    hf.line_chart(data, 'Gym Motivation', st)
    hf.line_chart(data, 'Healthy Eats', st)
    hf.line_chart(data, 'Mood of the Day', st)

def page1():
    st.markdown("<h1 style='text-align: center;'>Monthly Comparison</h1>", unsafe_allow_html=True)

    # Dropdown menu options
    options = {
        'Hours of Sleep': 'Cumulative Sleep Comparison',
        'Gym': 'Cumulative Gym Comparison',
        'Healthy Eats': 'Cumulative Healthy Eats Comparison',
        'Mood of the Day': 'Cumulative Mood of the Day Comparison'
    }

    # Create the dropdown menu
    selected_column = st.selectbox(
        "Select the metric to compare:",
        options.keys()
    )

    # Title dynamically based on selection
    st.subheader(f"{options[selected_column]}: This Month vs Last Month")

    # Generate and display the cumulative comparison line chart
    fig = hf.cumulative_last_30_days_comparison(data, selected_column)
    st.plotly_chart(fig)


def page2():
    st.markdown("<h1 style='text-align: center;'>Mood Drivers and Insights</h1>", unsafe_allow_html=True)
    # ----  Bar Chart Section ----

    # Select predictor variable from dropdown (excluding 'Mood of the Day' and non-useful columns)
    predictor_options = ['Caffeine Consumption', 'Hours of Sleep', 'Healthy Eats', 'Location', 'Gym', 'Alcohol Previous Day', 'Gym Previous Day', 'Vitamins Last 7 Days', 'Creatine Last 7 Days', 'Day of Week']
    selected_predictor = st.selectbox("Select a Predictor Variable:", predictor_options)

    # Add the dynamic subheading
    st.subheader(f"Mood of the Day by {selected_predictor}")

    # Generate and show the bar chart
    fig = hf.barchart_pred_vs_target(pred_column_name=selected_predictor, 
                                     target_column_name='Mood of the Day', 
                                     df=data)
    st.pyplot(fig)

    # Coefficient-based explanation
    st.header("What drove tomorrow's prediction?")
    feature_frame = hf.preprocess_for_logreg_manual(raw_data, numeric_columns, columns_to_drop)
    logreg_model, features = hf.load_logreg_manual_model()
    new_data = feature_frame.iloc[[-1]]
    contrib = hf.logreg_good_class_contributions(logreg_model, features, new_data)

    top_n = 10
    contrib_top = contrib.head(top_n)

    import plotly.express as px
    fig = px.bar(
        contrib_top.sort_values(),
        x=contrib_top.sort_values(),
        y=contrib_top.sort_values().index,
        orientation='h',
        color=contrib_top.sort_values(),
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0,
        labels={'x': 'Contribution (coef * value)', 'y': 'Feature'},
        title='Top contributors to good mood (today’s inputs)'
    )
    st.plotly_chart(fig, use_container_width=True)
# Page Selection Logic
if page == "Home":
    home()
elif page == "Monthly Comparison":
    page1()
elif page == "Mood Drivers and Insights":
    page2()
