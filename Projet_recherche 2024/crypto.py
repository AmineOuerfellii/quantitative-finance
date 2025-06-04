import asyncio
import websockets
import json
import numpy as np
import pandas as pd
import aiohttp
from scipy.stats import norm
from scipy.optimize import fsolve
import logging
from datetime import datetime, timedelta
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import webbrowser
from collections import deque
import time

# Configure logging (file and console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('option_pricing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global state
market_data = {}
positions = {}
volatility_surface = {}
price_history = deque(maxlen=1000)
delta_history = deque(maxlen=1000)
altcoin_data_cache = pd.DataFrame()

# Deribit live WebSocket URL
WS_URL = "wss://www.deribit.com/ws/api/v2"

# CoinGecko API base URL
COINGECKO_API = "https://api.coingecko.com/api/v3"

class VannaVolgaPricer:
    def __init__(self):
        self.r = 0.0

    def black_scholes_call(self, S, K, T, sigma):
        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        except Exception as e:
            logger.error(f"Black-Scholes error: {e}")
            return 0

    def vega(self, S, K, T, sigma):
        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            return S * np.sqrt(T) * norm.pdf(d1)
        except Exception as e:
            logger.error(f"Vega error: {e}")
            return 0

    def vanna(self, S, K, T, sigma):
        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return -norm.pdf(d1) * d2 / sigma
        except Exception as e:
            logger.error(f"Vanna error: {e}")
            return 0

    def volga(self, S, K, T, sigma):
        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S * np.sqrt(T) * norm.pdf(d1) * d1 * d2 / sigma
        except Exception as e:
            logger.error(f"Volga error: {e}")
            return 0

    def vanna_volga_price(self, S, K, T, atm_vol, rr_vol, bf_vol, market_price):
        try:
            sigma_atm = atm_vol
            sigma_rr = rr_vol
            sigma_bf = bf_vol
            K_atm = S
            K1 = S * np.exp(-0.1)
            K2 = S * np.exp(0.1)
            c_atm = self.black_scholes_call(S, K_atm, T, sigma_atm)
            c_k1 = self.black_scholes_call(S, K1, T, sigma_atm - sigma_rr / 2)
            c_k2 = self.black_scholes_call(S, K2, T, sigma_atm + sigma_rr / 2)
            vega_atm = self.vega(S, K_atm, T, sigma_atm)
            vega_k = self.vega(S, K, T, sigma_atm)
            vanna_atm = self.vanna(S, K_atm, T, sigma_atm)
            vanna_k = self.vanna(S, K, T, sigma_atm)
            volga_atm = self.volga(S, K_atm, T, sigma_atm)
            volga_k = self.volga(S, K, T, sigma_atm)
            x1 = vega_k / vega_atm if vega_atm != 0 else 0
            x2 = vanna_k / vanna_atm if vanna_atm != 0 else 0
            x3 = volga_k / volga_atm if volga_atm != 0 else 0
            market_adj = market_price - self.black_scholes_call(S, K, T, sigma_atm)
            vv_price = c_atm + x1 * (c_k1 + c_k2 - 2 * c_atm) + x2 * sigma_rr + x3 * sigma_bf
            price = max(vv_price + market_adj, 0)
            if price == 0:
                logger.warning("Vanna-Volga returned 0, falling back to Black-Scholes")
                price = self.black_scholes_call(S, K, T, sigma_atm)
            return price
        except Exception as e:
            logger.error(f"Vanna-Volga pricing error: {e}")
            return self.black_scholes_call(S, K, T, sigma_atm)

class DataFeed:
    async def fetch_historical_data(self, coin_id, days=30, retries=3, backoff=5):
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
                    async with session.get(url, timeout=10) as response:
                        if response.status == 429:
                            logger.warning(f"Rate limit hit for {coin_id}, retrying in {backoff} seconds")
                            await asyncio.sleep(backoff)
                            continue
                        data = await response.json()
                        prices = [entry[1] for entry in data["prices"]]
                        timestamps = [entry[0] / 1000 for entry in data["prices"]]
                        df = pd.DataFrame({'timestamp': timestamps, 'price': prices})
                        logger.info(f"Fetched historical data for {coin_id}: {len(df)} entries, Prices sample: {df['price'].head(5).tolist()}")
                        return df
            except Exception as e:
                logger.error(f"Error fetching CoinGecko data for {coin_id} (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(backoff)
        logger.error(f"Failed to fetch historical data for {coin_id} after {retries} attempts")
        return pd.DataFrame()

    def calculate_historical_volatility(self, prices, window=7):
        try:
            logger.info(f"Calculating rolling volatility with prices: {prices.head(5).tolist()}")
            returns = np.log(prices / prices.shift(1)).dropna()
            logger.info(f"Log returns: {returns.head(5).tolist()}")
            # Calculate rolling standard deviation over the window, annualize it
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            # Fill NaN values (first 'window' periods) with the first non-NaN volatility
            rolling_vol = rolling_vol.fillna(rolling_vol.dropna().iloc[0])
            logger.info(f"Rolling volatility sample: {rolling_vol.head(5).tolist()}")
            return rolling_vol
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return pd.Series([0.3] * len(prices), index=prices.index)

class PricingEngine:
    def __init__(self):
        self.vv_pricer = VannaVolgaPricer()
        self.time_to_maturity = {}

    def set_maturity(self, instrument, expiry_date):
        try:
            expiry = datetime.strptime(expiry_date, "%d%b%y")
            now = datetime.utcnow()
            t = max((expiry - now).days / 365.0, 1e-6)
            self.time_to_maturity[instrument] = t
        except Exception as e:
            logger.error(f"Error setting maturity for {instrument}: {e}")
            self.time_to_maturity[instrument] = 0.5

    async def price_option(self, instrument, S, K, market_price):
        try:
            logger.info(f"Pricing option {instrument}: S={S}, K={K}, market_price={market_price}")
            T = self.time_to_maturity.get(instrument, 0.5)
            atm_vol = market_data.get(instrument, {}).get("mark_iv", 50) / 100
            rr_vol = 0.05
            bf_vol = 0.02
            price = self.vv_pricer.vanna_volga_price(S, K, T, atm_vol, rr_vol, bf_vol, market_price)
            if price == 0:
                logger.warning(f"Price for {instrument} is 0, using fallback price")
                price = max(S * 0.1, 1000)
            price_history.append({
                'time': datetime.utcnow().timestamp(),
                'instrument': instrument,
                'price': price
            })
            logger.info(f"Priced {instrument}: {price}, Price history size: {len(price_history)}, Timestamp: {datetime.utcnow()}")
            return price
        except Exception as e:
            logger.error(f"Error pricing option {instrument}: {e}")
            return 1000

class OrderManager:
    async def update_orders(self, instrument, theo_price, spread=0.01, quantity=1):
        try:
            bid = theo_price * (1 - spread / 2)
            ask = theo_price * (1 + spread / 2)
            logger.info(f"Updating orders for {instrument}: Bid={bid}, Ask={ask}, Qty={quantity}")
        except Exception as e:
            logger.error(f"Error updating orders for {instrument}: {e}")

class Hedger:
    def __init__(self, pricing_engine):
        self.pricing_engine = pricing_engine

    async def delta_hedge(self):
        try:
            total_delta = 0
            logger.info(f"Starting delta hedge - Positions: {positions}, Market data keys: {list(market_data.keys())}")
            for instrument, qty in positions.items():
                if instrument in market_data:
                    data = market_data[instrument]
                    S = data.get("last_price", 50000)
                    K = float(instrument.split('-')[2])
                    T = self.pricing_engine.time_to_maturity.get(instrument, 0.5)
                    sigma = data.get("mark_iv", 50) / 100
                    logger.info(f"Calculating delta for {instrument}: S={S}, K={K}, T={T}, sigma={sigma}")
                    delta = self.pricing_engine.calculate_delta(S, K, T, sigma)
                    total_delta += qty * delta
                    logger.info(f"Delta for {instrument}: {delta}, Total delta: {total_delta}")
                else:
                    logger.warning(f"Instrument {instrument} not in market_data")
            hedge_amount = -total_delta
            logger.info(f"Hedging: Trade {hedge_amount} of underlying")
            if not delta_history:
                logger.warning("Delta history is empty, adding mock delta")
                delta_history.append({
                    'time': datetime.utcnow().timestamp(),
                    'delta': 0.5
                })
        except Exception as e:
            logger.error(f"Error in delta hedging: {e}")
            delta_history.append({
                'time': datetime.utcnow().timestamp(),
                'delta': 0.5
            })

class Backtester:
    def __init__(self, pricing_engine, data_feed):
        self.pricing_engine = pricing_engine
        self.data_feed = data_feed

    async def backtest(self, coin_id, days=30, K=50000, T=0.5):
        try:
            data = await self.data_feed.fetch_historical_data(coin_id, days)
            if data.empty:
                raise ValueError("No historical data")
            results = []
            for i in range(1, len(data)):
                S = data['price'].iloc[i]
                market_price = S * 0.1
                theo_price = await self.pricing_engine.price_option(f"{coin_id}-TEST", S, K, market_price)
                results.append({
                    "date": datetime.fromtimestamp(data['timestamp'].iloc[i]),
                    "spot": S,
                    "option_price": theo_price
                })
            results_df = pd.DataFrame(results)
            logger.info(f"Backtest results: {results_df.describe()}")
            return results_df
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return pd.DataFrame()

class Visualizer:
    def __init__(self):
        self.app = Dash(__name__)
        self.data_feed = DataFeed()
        self.setup_layout()

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("Crypto Option Pricing Dashboard"),
            dcc.Graph(id='price-plot'),
            dcc.Graph(id='delta-plot'),
            dcc.Graph(id='volatility-surface'),
            dcc.Graph(id='altcoin-volatility'),
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
        ])

        @self.app.callback(
            [Output('price-plot', 'figure'),
             Output('delta-plot', 'figure'),
             Output('volatility-surface', 'figure'),
             Output('altcoin-volatility', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_plots(n):
            try:
                price_df = pd.DataFrame(price_history)
                logger.info(f"Updating price plot, price history size: {len(price_history)}")
                if not price_df.empty:
                    logger.info(f"Price data sample: {price_df.tail(5).to_dict()}")
                    for instrument in price_df['instrument'].unique():
                        df = price_df[price_df['instrument'] == instrument]
                        times = [datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S') for t in df['time']]
                        logger.info(f"Timestamps for {instrument}: {times[:5]}")
                        logger.info(f"Prices for {instrument}: {df['price'].head(5).tolist()}")
                price_fig = go.Figure()
                if not price_df.empty:
                    for instrument in price_df['instrument'].unique():
                        df = price_df[price_df['instrument'] == instrument]
                        price_fig.add_trace(go.Scatter(
                            x=[datetime.fromtimestamp(t) for t in df['time']],
                            y=df['price'],
                            mode='lines+markers',
                            name=instrument
                        ))
                price_fig.update_layout(title="Real-Time Option Prices", xaxis_title="Time", yaxis_title="Price")

                delta_df = pd.DataFrame(delta_history)
                logger.info(f"Updating delta plot, delta history size: {len(delta_history)}")
                if not delta_df.empty:
                    logger.info(f"Delta data sample: {delta_df.tail(5).to_dict()}")
                delta_fig = go.Figure()
                if not delta_df.empty:
                    delta_fig.add_trace(go.Scatter(
                        x=[datetime.fromtimestamp(t) for t in delta_df['time']],
                        y=delta_df['delta'],
                        mode='lines+markers',
                        name='Portfolio Delta'
                    ))
                delta_fig.update_layout(title="Portfolio Delta", xaxis_title="Time", yaxis_title="Delta")

                strikes = np.linspace(40000, 60000, 10)
                maturities = np.linspace(0.1, 1.0, 5)
                S = market_data.get("BTC-PERPETUAL", {}).get("last_price", 50000)
                Z = [[0.5 for _ in strikes] for _ in maturities]
                vol_fig = go.Figure(data=[go.Surface(z=Z, x=strikes, y=maturities)])
                vol_fig.update_layout(title="Volatility Surface", scene=dict(
                    xaxis_title="Strike",
                    yaxis_title="Maturity",
                    zaxis_title="Implied Volatility"
                ))

                altcoin_fig = go.Figure()
                logger.info(f"Updating altcoin plot, altcoin data cache size: {len(altcoin_data_cache)}")
                if not altcoin_data_cache.empty:
                    rolling_vol = self.data_feed.calculate_historical_volatility(altcoin_data_cache['price'], window=7)
                    timestamps = [datetime.fromtimestamp(t) for t in altcoin_data_cache['timestamp']]
                    logger.info(f"XRP plot timestamps sample: {timestamps[:5]}")
                    logger.info(f"XRP rolling volatility values sample: {rolling_vol.iloc[:5].tolist()}")
                    altcoin_fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=rolling_vol,
                        mode='lines+markers',
                        name='XRP Volatility (7-Day Rolling)'
                    ))
                altcoin_fig.update_layout(
                    title="XRP Historical Volatility (7-Day Rolling)",
                    xaxis_title="Date",
                    yaxis_title="Volatility",
                    yaxis=dict(range=[0, max(0.2, rolling_vol.max() * 1.5)])
                )

                return price_fig, delta_fig, vol_fig, altcoin_fig
            except Exception as e:
                logger.error(f"Dash callback error: {e}")
                return go.Figure(), go.Figure(), go.Figure(), go.Figure()

    def run(self):
        logger.info("Starting Dash server on http://127.0.0.1:8050")
        webbrowser.open("http://127.0.0.1:8050")
        self.app.run_server(debug=False, port=8050)

async def connect_websocket():
    try:
        async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=10) as websocket:
            logger.info("Connected to Deribit WebSocket")
            instruments = [
                "ticker.BTC-27DEC24-50000-C.raw",
                "ticker.ETH-27DEC24-2500-C.raw",
                "ticker.BTC-PERPETUAL.raw",
                "ticker.ETH-PERPETUAL.raw"
            ]
            subscribe_msg = {
                "jsonrpc": "2.0",
                "method": "public/subscribe",
                "params": {"channels": instruments},
                "id": 1
            }
            await websocket.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to {instruments}")

            pricing_engine = PricingEngine()
            data_feed = DataFeed()
            order_manager = OrderManager()
            hedger = Hedger(pricing_engine)

            pricing_engine.set_maturity("BTC-27DEC24-50000-C", "27DEC24")
            pricing_engine.set_maturity("ETH-27DEC24-2500-C", "27DEC24")

            positions.update({
                "BTC-27DEC24-50000-C": 10,
                "ETH-27DEC24-2500-C": 5
            })

            mock_data = False
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=30)
                    data = json.loads(response)
                    if "params" in data and "data" in data["params"]:
                        instrument = data["params"]["data"]["instrument_name"]
                        market_data[instrument] = data["params"]["data"]
                        logger.info(f"Received data for {instrument}: {data['params']['data']}")
                        await process_market_data(pricing_engine, order_manager, hedger, data_feed)
                    else:
                        logger.warning(f"Unexpected WebSocket response: {data}")
                except asyncio.TimeoutError:
                    logger.warning("WebSocket receive timeout")
                    mock_data = True
                    break
                except websockets.exceptions.ConnectionClosed:
                    logger.error("WebSocket connection closed")
                    mock_data = True
                    break
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    mock_data = True
                    break

            if mock_data:
                logger.info("Using mock data due to WebSocket failure")
                for i in range(10):
                    market_data["BTC-27DEC24-50000-C"] = {
                        "last_price": 50000 + i * 100,
                        "mark_price": 51000 + i * 100,
                        "mark_iv": 50
                    }
                    market_data["BTC-PERPETUAL"] = {
                        "last_price": 50000 + i * 100
                    }
                    await process_market_data(pricing_engine, order_manager, hedger, data_feed)
                    await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Failed to connect to WebSocket: {e}")
        logger.info("Using mock data due to WebSocket connection failure")
        pricing_engine = PricingEngine()
        data_feed = DataFeed()
        order_manager = OrderManager()
        hedger = Hedger(pricing_engine)
        pricing_engine.set_maturity("BTC-27DEC24-50000-C", "27DEC24")
        positions.update({"BTC-27DEC24-50000-C": 10})
        for i in range(10):
            market_data["BTC-27DEC24-50000-C"] = {
                "last_price": 50000 + i * 100,
                "mark_price": 51000 + i * 100,
                "mark_iv": 50
            }
            market_data["BTC-PERPETUAL"] = {
                "last_price": 50000 + i * 100
            }
            await process_market_data(pricing_engine, order_manager, hedger, data_feed)
            await asyncio.sleep(1)

async def process_market_data(pricing_engine, order_manager, hedger, data_feed):
    global altcoin_data_cache
    try:
        logger.info(f"Processing market data: {list(market_data.keys())}")
        for instrument, data in market_data.items():
            if "PERPETUAL" not in instrument and "C" in instrument:
                S = market_data.get(f"{instrument.split('-')[0]}-PERPETUAL", {}).get("last_price", 50000)
                K = float(instrument.split('-')[2])
                market_price = data.get("mark_price", S * 0.1)
                logger.info(f"Pricing {instrument}: S={S}, K={K}, market_price={market_price}")
                theo_price = await pricing_engine.price_option(instrument, S, K, market_price)
                await order_manager.update_orders(instrument, theo_price)
        await hedger.delta_hedge()
        if int(datetime.utcnow().timestamp()) % 7200 == 0:
            altcoin_data_cache = await data_feed.fetch_historical_data("ripple")
            if not altcoin_data_cache.empty:
                vol = data_feed.calculate_historical_volatility(altcoin_data_cache['price'])
                S = altcoin_data_cache['price'].iloc[-1]
                K = S * 1.1
                T = 0.5
                altcoin_price = pricing_engine.vv_pricer.black_scholes_call(S, K, T, vol.iloc[-1])
                logger.info(f"XRP Option Price: {altcoin_price}")
    except Exception as e:
        logger.error(f"Error processing market data: {e}")

async def main():
    logger.info("Starting option pricing system")
    backtester = Backtester(PricingEngine(), DataFeed())
    backtest_results = await backtester.backtest("bitcoin", days=30, K=50000, T=0.5)
    if not backtest_results.empty:
        backtest_results.to_csv("backtest_results.csv")
        logger.info("Backtest results saved to backtest_results.csv")

    # Initial fetch of XRP data
    data_feed = DataFeed()
    global altcoin_data_cache
    altcoin_data_cache = await data_feed.fetch_historical_data("ripple")
    if altcoin_data_cache.empty:
        logger.warning("Initial XRP data fetch failed, using mock data")
        altcoin_data_cache = pd.DataFrame({
            'timestamp': [datetime.utcnow().timestamp() - i * 86400 for i in range(30)],
            'price': [1.0 * (1 + 0.1 * np.sin(i)) for i in range(30)]
        })

    # Force mock delta for testing
    for i in range(5):
        delta_history.append({
            'time': datetime.utcnow().timestamp() + i,
            'delta': 0.5 + i * 0.1
        })
    logger.info(f"Added mock delta data, delta history size: {len(delta_history)}")

    visualizer = Visualizer()
    websocket_task = asyncio.create_task(connect_websocket())
    visualizer.run()
    await websocket_task

if __name__ == "__main__":
    asyncio.run(main())