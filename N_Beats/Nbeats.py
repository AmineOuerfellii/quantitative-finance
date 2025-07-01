import streamlit as st
import plotly.graph_objs as go
import plotly.subplots as sp
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import econometron.Models.n_beats as nbeats


def fetch_stock_data(symbol, period="2y"):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        st.success(f"Successfully fetched {len(data)} days of data for {symbol}")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def prepare_stock_data(data, target_column='Close'):
    prices = data[target_column].values
    prices = prices[~np.isnan(prices)]
    st.info(f"Prepared {len(prices)} data points for forecasting")
    return prices

def plot_data_overview(data, symbol):
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"{symbol} - Closing Price",
            f"{symbol} - Trading Volume",
            f"{symbol} - Daily Returns",
            f"{symbol} - Return Distribution"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Volume'], mode='lines', name='Volume', line=dict(color='green', width=2)),
        row=1, col=2
    )
    
    daily_returns = data['Close'].pct_change().dropna()
    fig.add_trace(
        go.Scatter(x=data.index[1:], y=daily_returns, mode='lines', name='Daily Returns', line=dict(color='red', width=2)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=daily_returns, nbinsx=50, name='Return Distribution', marker=dict(color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        width=1000,
        showlegend=False,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Daily Return", row=2, col=2)
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=1, col=2)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    return fig

def plot_training_history_plotly(history):
    """Custom function to create a Plotly figure for training history"""
    fig = sp.make_subplots(
        rows=2, cols=1,
        subplot_titles=("Training and Validation Loss", "Learning Rate"),
        vertical_spacing=0.15
    )
    
    # Assuming history contains 'train_loss' and 'val_loss' lists
    epochs = list(range(1, len(history.get('train_loss', [])) + 1))
    
    fig.add_trace(
        go.Scatter(x=epochs, y=history.get('train_loss', []), mode='lines', name='Training Loss', line=dict(color='blue')),
        row=1, col=1
    )
    if 'val_loss' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history.get('val_loss', []), mode='lines', name='Validation Loss', line=dict(color='red')),
            row=1, col=1
        )
    
    if 'lr' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history.get('lr', []), mode='lines', name='Learning Rate', line=dict(color='green')),
            row=2, col=1
        )
    
    fig.update_layout(
        height=500,
        width=1000,
        showlegend=True,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Learning Rate", row=2, col=1)
    
    return fig

def main_example(symbol, period, stack_configs, training_params):
    st.write("=" * 60)
    st.write("N-BEATS Stock Price Forecasting Example")
    st.write("=" * 60)
    
    st.write(f"\n1. Fetching data for {symbol}...")
    stock_data = fetch_stock_data(symbol, period)
    if stock_data is None:
        return None, None, None, f"Failed to fetch data for {symbol}"
    
    data_info = {
        'range': f"{stock_data.index[0].date()} to {stock_data.index[-1].date()}",
        'latest_price': stock_data['Close'].iloc[-1],
        'data_points': len(stock_data)
    }
    
    st.write(f"Data range: {data_info['range']}")
    st.write(f"Latest closing price: ${data_info['latest_price']:.2f}")
    
    st.write("\n2. Plotting data overview...")
    overview_fig = plot_data_overview(stock_data, symbol)
    st.plotly_chart(overview_fig,key="chart1",use_container_width=True)
    st.write("\n3. Preparing data for forecasting...")
    prices = prepare_stock_data(stock_data, target_column='Close')
    
    st.write("\n4. Creating N-BEATS model...")
    backcast_length = training_params['backcast_length']
    forecast_length = training_params['forecast_length']
    
    model = nbeats.NeuralForecast(
        stack_configs=stack_configs,
        backcast_length=backcast_length,
        forecast_length=forecast_length
    )
    
    st.write(f"Model created with {sum(p.numel() for p in model.model.parameters()):,} parameters")
    
    st.write("\n5. Processing and splitting data...")
    train_data, val_data, test_data = model.process_data(
        data=prices,
        train_ratio=0.7,
        val_ratio=0.15,
        normalize=False
    )
    
    st.write("\n6. Training the model...")
    with st.spinner("Training model... This may take a while."):
        history = model.fit(
            train_data=train_data,
            val_data=val_data,
            epochs=training_params['epochs'],
            batch_size=training_params['batch_size'],
            learning_rate=training_params['learning_rate'],
            optimizer=training_params['optimizer'],
            loss_function='mae',
            early_stopping=True,
            patience=15,
            scheduler='plateau',
            gradient_clip=training_params['gradient_clip'],
            verbose=True
        )

    st.success("Model training completed successfully!")
    st.write("\n7. Plotting training history...")
    train_fig = plot_training_history_plotly(history)
    st.plotly_chart(train_fig,key="chart2",use_container_width=True)

    st.write("\n8. Evaluating on test data...")
    test_metrics = model.evaluate(
        test_data=test_data,
        metrics=['mae', 'mse', 'rmse', 'mape']
    )
    
    st.write("\n9. Generating forecasts...")
    recent_data = prices[-100:]
    input_sequence = recent_data[-backcast_length:]
    forecast, components = model.forecast(
        input_sequence=input_sequence,
        return_components=True
    )
    
    st.write(f"Forecast for next {forecast_length} days:")
    forecast_text = [f"Day {i+1}: ${pred:.2f}" for i, pred in enumerate(forecast.flatten())]
    
    st.write("\n10. Plotting forecast results...")
    hist_data = prices[-60:]
    hist_time = np.arange(len(hist_data))
    forecast_flat = forecast.flatten()
    forecast_time = np.arange(len(hist_data), len(hist_data) + len(forecast_flat))
    
    forecast_fig = sp.make_subplots(rows=2, cols=1, subplot_titles=(
        f'{symbol} Stock Price Forecast - N-BEATS Model',
        'Forecast Components Breakdown'
    ))
    
    forecast_fig.add_trace(
        go.Scatter(x=hist_time, y=hist_data, mode='lines', name='Historical Prices', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    forecast_fig.add_trace(
        go.Scatter(x=forecast_time, y=forecast_flat, mode='lines+markers', name='N-BEATS Forecast', line=dict(color='red', width=2), marker=dict(symbol='circle')),
        row=1, col=1
    )
    
    forecast_fig.add_vrect(
        x0=len(hist_data) - backcast_length, x1=len(hist_data)-1, fillcolor="green", opacity=0.2,
        layer="below", line_width=0, annotation_text="Input Sequence"
    )
    forecast_fig.add_vline(x=len(hist_data)-1, line_dash="dash", line_color="black")
    
    colors = ['purple', 'orange', 'brown', 'pink', 'gray']
    for i, (name, component) in enumerate(components.items()):
        forecast_fig.add_trace(
            go.Scatter(x=forecast_time, y=component.flatten(), mode='lines', name=name, line=dict(color=colors[i % len(colors)], width=1.5)),
            row=2, col=1
        )
    
    forecast_fig.update_layout(
        height=600,
        width=1000,
        showlegend=True,
        template='plotly_white'
    )
    
    forecast_fig.update_xaxes(title_text="Days", row=1, col=1)
    forecast_fig.update_xaxes(title_text="Days", row=2, col=1)
    forecast_fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    forecast_fig.update_yaxes(title_text="Price Contribution ($)", row=2, col=1)
    
    st.write("\n11. Model Summary:")
    st.write("=" * 40)
    summary = model.get_model_summary()
    for key, value in summary.items():
        if key != 'model_info':
            st.write(f"{key}: {value}")
    
    model_path = f"nbeats_{symbol.lower()}_model.pth"
    model.save_model(model_path)
    st.success(f"Model saved to: {model_path}")
    
    return model, test_metrics, forecast, data_info, overview_fig, train_fig, forecast_fig, forecast_text

def hyperparameter_example(symbol="GOOGL", period="1y"):
    st.write("\n" + "="*60)
    st.write("HYPERPARAMETER TUNING EXAMPLE")
    st.write("="*60)
    
    stock_data = fetch_stock_data(symbol, period)
    prices = prepare_stock_data(stock_data)
    
    stack_configs = [
        {
            'n_blocks': 2,
            'basis_type': 'generic',
            'n_layers_per_block': 3,
            'hidden_size': 128,
            'share_weights': True
        }
    ]
    
    model = nbeats.NeuralForecast(
        stack_configs=stack_configs,
        backcast_length=20,
        forecast_length=3
    )
    
    train_data, val_data, test_data = model.process_data(
        data=prices,
        train_ratio=0.8,
        val_ratio=0.15,
        normalize=False
    )
    
    param_grid = {
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'batch_size': [16, 32, 64],
        'optimizer': ['adam', 'adamw']
    }
    
    with st.spinner("Running hyperparameter tuning..."):
        results = model.hyperparameter_finder(
            train_data=train_data,
            val_data=val_data,
            param_grid=param_grid,
            max_trials=6,
            epochs=30
        )
    
    return results

def create_trading_strategy(model, prices, lookback=30, threshold=0.02):
    positions = []
    signals = []
    
    for i in range(lookback, len(prices) - 5):
        input_seq = prices[i-lookback:i]
        forecast = model.forecast(input_seq)
        current_price = prices[i]
        future_price = forecast[0, 0]
        expected_return = (future_price - current_price) / current_price
        
        if expected_return > threshold:
            signal = 1
        elif expected_return < -threshold:
            signal = -1
        else:
            signal = 0
        
        signals.append(signal)
        positions.append(expected_return)
    
    return signals, positions

def main():
    st.set_page_config(
        page_title="N-BEATS Stock Forecasting",
        page_icon="📈",
        layout="wide"
    )
    
    st.title('📈 N-BEATS Stock Price Forecasting Dashboard')
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        
        stock_symbol = st.text_input("Stock Symbol", value="AAPL")
        
        time_period = st.selectbox(
            "Time Period",
            options=["1y", "2y", "5y"],
            index=1
        )
        
        analysis_type = st.selectbox(
            "Analysis Type",
            options=["Main Forecast", "Hyperparameter Tuning"],
            index=0
        )
        
        # Stack configuration
        st.subheader("N-BEATS Stack Configuration")
        num_stacks = st.number_input("Number of Stacks", min_value=1, max_value=5, value=3)
        
        stack_configs = []
        for i in range(num_stacks):
            with st.expander(f"Stack {i+1}"):
                n_blocks = st.number_input(f"Number of Blocks (Stack {i+1})", min_value=1, max_value=5, value=3, key=f"n_blocks_{i}")
                basis_type = st.selectbox(f"Basis Type (Stack {i+1})", options=['polynomial', 'fourier', 'generic'], index=2, key=f"basis_type_{i}")
                n_layers = st.number_input(f"Layers per Block (Stack {i+1})", min_value=1, max_value=10, value=4, key=f"n_layers_{i}")
                hidden_size = st.number_input(f"Hidden Size (Stack {i+1})", min_value=64, max_value=1024, value=512, step=64, key=f"hidden_size_{i}")
                degree = st.number_input(f"Degree (Stack {i+1})", min_value=1, max_value=5, value=1 if basis_type == 'polynomial' else 3, key=f"degree_{i}")
                share_weights = st.checkbox(f"Share Weights (Stack {i+1})", value=True, key=f"share_weights_{i}")
                
                stack_configs.append({
                    'n_blocks': n_blocks,
                    'basis_type': basis_type,
                    'n_layers_per_block': n_layers,
                    'hidden_size': hidden_size,
                    'degree': degree,
                    'share_weights': share_weights
                })
        
        # Training parameters
        st.subheader("Training Parameters")
        backcast_length = st.number_input("Backcast Length", min_value=5, max_value=100, value=30)
        forecast_length = st.number_input("Forecast Length", min_value=1, max_value=20, value=5)
        epochs = st.number_input("Epochs", min_value=10, max_value=500, value=100)
        batch_size = st.number_input("Batch Size", min_value=8, max_value=256, value=64, step=8)
        learning_rate = st.selectbox("Learning Rate", options=[1e-4, 1e-3, 1e-2, 5e-2], index=1)
        optimizer = st.selectbox("Optimizer", options=['sgd', 'adam', 'adamw'], index=2)
        gradient_clip = st.number_input("Gradient Clipping", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        training_params = {
            'backcast_length': backcast_length,
            'forecast_length': forecast_length,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'gradient_clip': gradient_clip
        }
        
        run_analysis = st.button("🚀 Run Analysis", type="primary")
    
    # Main content
    if run_analysis:
        if analysis_type == "Main Forecast":
            try:
                result = main_example(stock_symbol, time_period, stack_configs, training_params)
                if result[0] is None:
                    st.error(result[3])
                else:
                    model, test_metrics, forecast, data_info, overview_fig, train_fig, forecast_fig, forecast_text = result
                    
                    # Display data info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Data Range", data_info['range'])
                    with col2:
                        st.metric("Latest Price", f"${data_info['latest_price']:.2f}")
                    with col3:
                        st.metric("Data Points", data_info['data_points'])
                    
                    # Display plots
                    st.subheader("📊 Data Overview")
                    st.plotly_chart(overview_fig, key="chart4", use_container_width=True)

                    st.subheader("📈 Training History")
                    st.plotly_chart(train_fig, key="chart5", use_container_width=True)

                    st.subheader("🔮 Forecast Results")
                    st.plotly_chart(forecast_fig, key="chart6", use_container_width=True)

                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🎯 Forecast Predictions")
                        for text in forecast_text:
                            st.write(f"• {text}")
                    
                    with col2:
                        st.subheader("📊 Test Metrics")
                        for key, value in test_metrics.items():
                            st.write(f"• {key}: {value:.4f}")
                            
            except Exception as e:
                st.error(f"Error running main analysis: {e}")
        
        elif analysis_type == "Hyperparameter Tuning":
            try:
                results = hyperparameter_example(stock_symbol, time_period)
                if results is None:
                    st.error(f"Failed to fetch data for {stock_symbol}")
                else:
                    st.subheader("🔧 Hyperparameter Tuning Results")
                    st.write(f"**Best parameters:** {results['best_params']}")
                    st.write(f"**Best validation score:** {results['best_score']:.6f}")
                    
            except Exception as e:
                st.error(f"Error running hyperparameter tuning: {e}")
    
    else:
        st.info("👈 Configure your analysis settings in the sidebar and click 'Run Analysis' to get started!")
        
        # Display some information about the app
        st.markdown("""
        ## About N-BEATS Stock Forecasting
        
        This dashboard implements the N-BEATS (Neural Basis Expansion Analysis for Time Series) model for stock price forecasting.
        
        ### Features:
        - **Main Forecast**: Comprehensive stock analysis with price predictions and customizable model architecture
        - **Hyperparameter Tuning**: Optimize model parameters for better performance  
        
        ### How to use:
        1. Enter a stock symbol (e.g., AAPL, GOOGL, TSLA)
        2. Select the time period for historical data
        3. Configure N-BEATS stack parameters and training settings
        4. Choose the type of analysis
        5. Click "Run Analysis" to start
        
        """)

if __name__ == '__main__':
    main()