# plot_utils.py

import numpy as np
import plotly.graph_objects as go

def plot_similarity_scores(chunks, distances):
    similarity_scores = 1 / (1 + np.array(distances))
    max_score = max(similarity_scores)

    fig = go.Figure(data=[go.Bar(
        x=[f"Chunk {i+1}" for i in range(len(chunks))],
        y=similarity_scores,
        text=[f"{score:.2f}" for score in similarity_scores],
        textposition='auto',
        marker_color='rgb(53, 133, 93)',
    )])

    fig.update_layout(
        yaxis_title="Similarity Score",
        yaxis_range=[0, max_score * 1.1],
        template="plotly_dark",
        height=150,
        margin=dict(l=0, r=10, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def plot_confidence_gauge(confidence: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Response Confidence", 'font': {'size': 16}},
        number={'font': {'size': 30}, 'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'size': 14}},
            'bar': {'color': "darkgreen"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "rgba(255,255,255,0.1)"},
                {'range': [50, 80], 'color': "rgba(255,255,255,0.3)"},
                {'range': [80, 100], 'color': "rgba(255,255,255,0.5)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'size': 14}
    )

    return fig

def get_confidence_legend(confidence: float) -> str:
    if confidence >= 0.7:
        return "High confidence: The response is likely accurate and well-supported by the data."
    elif confidence >= 0.5:
        return "Moderate confidence: The response is generally reliable but may have some uncertainties."
    elif confidence >= 0.3:
        return "Low confidence: The response may be speculative or based on limited information."
    else:
        return "Very low confidence: The response should be treated as highly uncertain."

# End of plot_utils.py
