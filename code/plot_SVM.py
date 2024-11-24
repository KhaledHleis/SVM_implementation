import plotly.graph_objects as go
import numpy as np

def plot_svm_decision_boundary(X, y, w, b):
    """
    Plot the SVM data and decision boundary using Plotly.

    Parameters:
    - X: ndarray, shape (n_samples, n_features), input data.
    - y: ndarray, shape (n_samples,), labels (+1 or -1).
    - w: ndarray, shape (n_features,), weight vector.
    - b: float, bias term.
    """
    # Prepare data points for plotting
    trace_pos = go.Scatter(
        x=X[y == 1, 0], y=X[y == 1, 1],
        mode='markers', marker=dict(color='blue', symbol='circle'),
        name="Class +1"
    )
    
    trace_neg = go.Scatter(
        x=X[y == -1, 0], y=X[y == -1, 1],
        mode='markers', marker=dict(color='red', symbol='x'),
        name="Class -1"
    )
    
    # Plot decision boundary
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = -(w[0] * x1 + b) / w[1]
    
    boundary = go.Scatter(
        x=x1, y=x2, mode='lines', line=dict(color='black'),
        name="Decision Boundary"
    )
    
    # Plot margins
    x2_margin_pos = -(w[0] * x1 + b - 1) / w[1]
    x2_margin_neg = -(w[0] * x1 + b + 1) / w[1]
    
    margin_pos = go.Scatter(
        x=x1, y=x2_margin_pos, mode='lines', line=dict(color='black', dash='dash'),
        name="Margin"
    )
    
    margin_neg = go.Scatter(
        x=x1, y=x2_margin_neg, mode='lines', line=dict(color='black', dash='dash'),
        name="Margin"
    )
    
    # Create the layout
    layout = go.Layout(
        title="SVM Decision Boundary",
        xaxis=dict(title='Feature 1'),
        yaxis=dict(title='Feature 2'),
        showlegend=True,
        plot_bgcolor='white',
        height=600, width=800
    )
    
    # Create the figure
    fig = go.Figure(data=[trace_pos, trace_neg, boundary, margin_pos, margin_neg], layout=layout)
    
    # Show the plot
    fig.show()

# Plot convergence of the cost function using Plotly
def plot_convergence(costs):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(costs) + 1), y=costs, mode="lines+markers", name="Cost"
        )
    )
    fig.update_xaxes(type="log", title_text="Iteration")
    fig.update_yaxes(type="log", title_text="Cost")
    fig.update_layout(title="Convergence of Cost Function", width=700, height=500)
    fig.show()


# Plot the gradient norms using Plotly
def plot_gradient_norms(gradient_norms):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(gradient_norms) + 1),
            y=gradient_norms,
            mode="lines+markers",
            name="Gradient Norm",
        )
    )
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Gradient Norm")
    fig.update_layout(title="Gradient Norms over Iterations", width=700, height=500)
    fig.show()
