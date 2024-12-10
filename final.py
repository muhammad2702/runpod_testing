import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from datetime import datetime
import numpy as np
from ta import trend, momentum, volatility
from sklearn.preprocessing import LabelEncoder
import pickle
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.init import xavier_uniform_ as xavier_uniform
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score  # If you want to use balanced accuracy
from torch.nn.init import xavier_uniform_
from datetime import datetime, timedelta
import runpod
import joblib

# Configuration
API_KEY = 'de_kgSuhw6v4KnRK0wprJCoBAIhqSd5R'  # Replace with your actual API key
BASE_URL = 'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}'

# List of cryptocurrencies (tickers) you want to collect data for
TICKERS = [
    'X:AAVEUSD',
    'X:AVAXUSD',
    'X:BATUSD',
    'X:LINKUSD',
    'X:UNIUSD',
    'X:SUSHIUSD',
    'X:PNGUSD',
    'X:JOEUSD',
    'X:XAVAUSD',
    'X:ATOMUSD',
    'X:ALGOUSD',
    'X:ARBUSD',
    'X:1INCHUSD',
    'X:DAIUSD',
    # Add more tickers as needed
]

# Timeframes you want to collect data for
TIMEFRAMES = [
    {'multiplier': 1, 'timespan': 'second'},
]

# Date range for data collection

# Directory to save the collected data
DATA_DIR = 'crypto_data'
os.makedirs(DATA_DIR, exist_ok=True)

def daterange(start_date, end_date, delta):
    current = start_date
    while current <= end_date:
        yield current
        current += delta

def fetch_data(ticker, multiplier, timespan, from_date, to_date):
    url = BASE_URL.format(
        ticker=ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_date=from_date.strftime('%Y-%m-%d'),
        to_date=to_date.strftime('%Y-%m-%d')
    )
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': '50000',  # Maximum allowed by Polygon.io
        'apiKey': API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('results', [])
    else:
        print(f"Error fetching data for {ticker} - {timespan}: {response.status_code} - {response.text}")
        return []

def collect(START_DATE, END_DATE):
    start = datetime.strptime(START_DATE, '%Y-%m-%d')
    end = datetime.strptime(END_DATE, '%Y-%m-%d')

    for ticker in TICKERS:
        ticker_dir = os.path.join(DATA_DIR, ticker.replace(":", "_"))
        os.makedirs(ticker_dir, exist_ok=True)

        for timeframe in TIMEFRAMES:
            multiplier = timeframe['multiplier']
            timespan = timeframe['timespan']
            filename = f"{ticker.replace(':', '_')}_{multiplier}{timespan}.csv"
            filepath = os.path.join(ticker_dir, filename)

            print(f"Fetching data for {ticker} - {multiplier}{timespan}")

            # To handle large date ranges, you might need to split the requests
            # For simplicity, we'll attempt to fetch all data at once
            data = fetch_data(ticker, multiplier, timespan, start, end)

            if data:
                df = pd.DataFrame(data)
                # Convert timestamp to datetime
                df['t'] = pd.to_datetime(df['t'], unit='ms')
                # Save to CSV
                df.to_csv(filepath, index=False)
                print(f"Saved {len(df)} records to {filepath}")
            else:
                print(f"No data fetched for {ticker} - {multiplier}{timespan}")

            # Respect API rate limits
            time.sleep(1)  # Adjust sleep time based on your rate limits

class CryptoDataPreprocessor:
    def __init__(self, raw_data_dir='crypto_data', preprocessed_data_dir='preprocessed_data', columns_to_add=None):
        """
        Initializes the CryptoDataPreprocessor.

        :param raw_data_dir: Directory containing raw CSV data.
        :param preprocessed_data_dir: Directory to save preprocessed data.
        :param columns_to_add: List of columns to include in the final output.
        """
        self.raw_data_dir = raw_data_dir
        self.preprocessed_data_dir = preprocessed_data_dir
        self.columns_to_add = columns_to_add or ['leg_direction', 'close_price']  # Default columns
        os.makedirs(self.preprocessed_data_dir, exist_ok=True)
        self.label_encoders = {}

    def preprocess_file(self, df):
        """
        Applies preprocessing steps to the DataFrame.

        :param df: pandas DataFrame with raw data.
        :return: Preprocessed DataFrame and label encoders.
        """
        required_columns = ['c', 'h', 'l']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Compute RSI
        rsi = momentum.RSIIndicator(close=df['c'], window=14)
        df['RSI'] = rsi.rsi()

        # Compute MACD with smoothing
        macd = trend.MACD(close=df['c'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd().rolling(window=3).mean()
        df['MACD_signal'] = macd.macd_signal().rolling(window=3).mean()
        df['MACD_diff'] = macd.macd_diff().rolling(window=3).mean()

        # Compute ATR
        atr = volatility.AverageTrueRange(high=df['h'], low=df['l'], close=df['c'], window=14)
        df['ATR'] = atr.average_true_range().rolling(window=3).mean()

        # Compute Bollinger Bands Width
        bollinger = volatility.BollingerBands(close=df['c'], window=20, window_dev=2)
        df['BB_upper'] = bollinger.bollinger_hband().rolling(window=3).mean()
        df['BB_lower'] = bollinger.bollinger_lband().rolling(window=3).mean()
        df['BB_width'] = ((df['BB_upper'] - df['BB_lower']) / df['c']).rolling(window=3).mean()

        # Compute ADX with smoothing
        adx = trend.ADXIndicator(high=df['h'], low=df['l'], close=df['c'], window=14)
        df['ADX'] = adx.adx().rolling(window=3).mean()

        # Drop initial rows with NaN values
        df.dropna(inplace=True)

        # Classify Market Environments
        df, label_encoders = self.classify_market_environments(df)

        # Calculate Leg Data
        df = self.calculate_leg_data(df)

        # Classify Percent Change
        df = self.classify_percent_change(df)
        df['close_price'] = df['c']  # Assuming 'c' is the closing price

        return df, label_encoders

    def classify_market_environments(self, df):
        """
        Classify market environments into numerical categories for model training.

        :param df: pandas DataFrame with technical indicators.
        :return: DataFrame with new classification columns and label encoders.
        """
        df['ATR_mavg'] = df['ATR'].rolling(window=14).mean()
        df['ATR_vol'] = np.where(df['ATR'] > 1.2 * df['ATR_mavg'], 'h',
                                 np.where(df['ATR'] < 0.8 * df['ATR_mavg'], 'l', 'Medium'))

        df['BB_mavg'] = df['BB_width'].rolling(window=20).mean()
        df['BB_vol'] = np.where(df['BB_width'] > 1.2 * df['BB_mavg'], 'h',
                                np.where(df['BB_width'] < 0.8 * df['BB_mavg'], 'l', 'Medium'))

        df['daily_return'] = df['c'].pct_change()
        df['RV'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)
        rv_80 = df['RV'].quantile(0.8)
        rv_20 = df['RV'].quantile(0.2)
        df['RV_vol'] = np.where(df['RV'] > rv_80, 'h',
                                np.where(df['RV'] < rv_20, 'l', 'Medium'))

        df['Volatility'] = df[['ATR_vol', 'BB_vol', 'RV_vol']].mode(axis=1)[0]

        df['Trend'] = np.where(
            (df['MACD'] > df['MACD_signal']) & (df['MACD'] > 0), 'Bullish',
            np.where(
                (df['MACD'] < df['MACD_signal']) & (df['MACD'] < 0), 'Bearish', 'Neutral'
            )
        )

        df['Trend_strength'] = np.where(df['ADX'] > 25, 'Strong', 'Weak')

        df['Market_Environment'] = df.apply(
            lambda row: f"{row['Volatility']} Vol/{row['Trend']}" if row['Trend_strength'] == 'Strong' else f"{row['Volatility']} Vol/Neutral",
            axis=1
        )

        label_encoders = {}
        categorical_columns = ['ATR_vol', 'BB_vol', 'RV_vol', 'Volatility', 'Trend', 'Trend_strength', 'Market_Environment']
        for column in categorical_columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le

        self.label_encoders = label_encoders
        return df, label_encoders

    def calculate_leg_data(self, df):
        df['percent_delta'] = df['c'].pct_change()
        df = df.reset_index(drop=True)

        previous_leg_change = 0
        previous_leg_length = 0
        current_leg_change = 0
        current_leg_length = 0
        current_direction = None

        previous_changes = []
        previous_lengths = []
        current_changes = []
        current_lengths = []
        leg_directions = []

        for i in range(len(df)):
            if i == 0:
                current_leg_change = 0
                current_leg_length = 0
                current_direction = 0  # Neutral at start
            else:
                percent_delta = df.at[i, 'percent_delta']
                if current_leg_length == 0:
                    current_leg_change = percent_delta
                    current_leg_length = 1
                    current_direction = 1 if percent_delta > 0 else 0
                else:
                    if (current_leg_change > 0 and percent_delta > 0) or (current_leg_change < 0 and percent_delta < 0):
                        current_leg_change += percent_delta
                        current_leg_length += 1
                    else:
                        previous_leg_change = current_leg_change
                        previous_leg_length = current_leg_length
                        current_leg_change = percent_delta
                        current_leg_length = 1
                        current_direction = 1 if percent_delta > 0 else 0
            previous_changes.append(previous_leg_change)
            previous_lengths.append(previous_leg_length)
            current_changes.append(current_leg_change)
            current_lengths.append(current_leg_length)
            leg_directions.append(current_direction)

        df['previous_leg_change'] = previous_changes
        df['previous_leg_length'] = previous_lengths
        df['current_leg_change'] = current_changes
        df['current_leg_length'] = current_lengths
        df['leg_direction'] = leg_directions

        df.drop(columns=['percent_delta'], inplace=True)
        return df

    def classify_percent_change(self, df):
        df['percent_change'] = df['c'].pct_change()
        df.dropna(inplace=True)

        percentiles = df['percent_change'].quantile([0.05, 0.20, 0.40, 0.60, 0.80, 0.95]).to_dict()

        def classify(x, p):
            if x < p[0.05]:
                return 'Down a Lot'
            elif x < p[0.20]:
                return 'Down Moderate'
            elif x < p[0.40]:
                return 'Down a Little'
            elif x < p[0.60]:
                return 'No Change'
            elif x < p[0.80]:
                return 'Up a Little'
            elif x < p[0.95]:
                return 'Up Moderate'
            else:
                return 'Up a Lot'

        df['percent_change_classification'] = df['percent_change'].apply(lambda x: classify(x, percentiles))

        le = LabelEncoder()
        df['percent_change_classification'] = le.fit_transform(df['percent_change_classification'].astype(str))
        self.label_encoders['percent_change_classification'] = le

        return df

    def save_preprocessed_data(self, df, filepath):
        """
        Saves the preprocessed DataFrame to a CSV file with selected columns.

        :param df: pandas DataFrame with preprocessed data.
        :param filepath: Path where the CSV will be saved.
        """
        df[self.columns_to_add].to_csv(filepath, index=False)
        print(f"Saved preprocessed data to {filepath}")

    def preprocess_all_files(self):
        # Traverse the raw_data_dir
        processed_tickers = []
        for ticker in tqdm(os.listdir(self.raw_data_dir), desc='Processing Tickers'):
            ticker_raw_dir = os.path.join(self.raw_data_dir, ticker)
            ticker_preprocessed_dir = os.path.join(self.preprocessed_data_dir, ticker)
            
            # Create or clean the preprocessed directory
            if os.path.exists(ticker_preprocessed_dir):
                # Remove all existing files in the preprocessed directory
                for old_file in os.listdir(ticker_preprocessed_dir):
                    old_file_path = os.path.join(ticker_preprocessed_dir, old_file)
                    os.remove(old_file_path)
            else:
                os.makedirs(ticker_preprocessed_dir, exist_ok=True)

            for file in tqdm(os.listdir(ticker_raw_dir), desc=f'Processing {ticker}', leave=False):
                if file.endswith('.csv'):
                    raw_filepath = os.path.join(ticker_raw_dir, file)
                    preprocessed_filename = file.replace('.csv', '_preprocessed.csv')
                    preprocessed_filepath = os.path.join(ticker_preprocessed_dir, preprocessed_filename)

                    # Read raw CSV
                    try:
                        df_raw = pd.read_csv(raw_filepath)
                        # Ensure timestamp is datetime if needed
                        if 'timestamp' in df_raw.columns and not pd.api.types.is_datetime64_any_dtype(df_raw['timestamp']):
                            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
                    except Exception as e:
                        print(f"Error reading {raw_filepath}: {e}")
                        continue

                    # Preprocess
                    try:
                        df_preprocessed, label_encoders = self.preprocess_file(df_raw)  # Unpack the tuple
                    except Exception as e:
                        print(f"Error preprocessing {raw_filepath}: {e}")
                        continue

                    # Print head of the DataFrame
                    print(f"Preprocessed data for {file}:")
                    print(df_preprocessed.head())  # Print the entire head without truncation

                    # Save preprocessed data
                    self.save_preprocessed_data(df_preprocessed, preprocessed_filepath)

                    print(f"Saved preprocessed file to {preprocessed_filepath}")

        processed_tickers_count = len(set(processed_tickers))
        with open('processed_tickers_count.txt', 'w') as f:
            f.write(str(processed_tickers_count))
        print(f"Processed {processed_tickers_count} tickers and saved the count.")

def preprocess():
    """
    Main function to preprocess cryptocurrency data.
    """
    # Define directories
    raw_data_dir = 'crypto_data'
    preprocessed_data_dir = 'preprocessed_data'

    # Columns to include in the final output
    columns_to_add = [
        'leg_direction', 'close_price', 'o', 'l', 'h', 't', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
        'ATR', 'BB_width', 'ADX',  # , 'Volatility', 'Trend', 'Trend_strength',
        'Market_Environment', 'percent_change_classification'  # 'previous_leg_change', 'previous_leg_length',
        # 'current_leg_change', 'current_leg_length'
    ]

    # Initialize the preprocessor
    preprocessor = CryptoDataPreprocessor(
        raw_data_dir=raw_data_dir,
        preprocessed_data_dir=preprocessed_data_dir,
        columns_to_add=columns_to_add
    )

    # Process all files
    print("Starting preprocessing...")
    preprocessor.preprocess_all_files()
    print("Preprocessing completed.")

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor of shape [num_classes], or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size]
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Gather the log probability of the target class only
        batch_indices = torch.arange(len(targets), dtype=torch.long, device=targets.device)
        pt = probs[batch_indices, targets]       # shape: [batch_size]
        log_pt = log_probs[batch_indices, targets]  # shape: [batch_size]

        # Apply alpha if provided
        if self.alpha is not None:
            # alpha is [num_classes], select alpha for each target
            at = self.alpha[targets]  # shape: [batch_size]
        else:
            at = 1.0

        # Compute focal loss
        # FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
        focal_loss = -at * ((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ShortTermTransformerModel(nn.Module):
    def __init__(self, num_features, num_cryptos, d_model=64, nhead=2, num_encoder_layers=1,
                 dim_feedforward=64, dropout=0.1, num_classes=7, max_seq_length=50):
        super(ShortTermTransformerModel, self).__init__()
        self.d_model = d_model
        self.crypto_embedding = nn.Embedding(num_cryptos, d_model)
        
        self.input_linear = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.PReLU(),
            nn.LayerNorm(d_model)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.percent_change_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        self.leg_direction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )

        self.initialize_weights()

    def forward(self, src, crypto_id):
        # Transform input features and add crypto embedding
        src = self.input_linear(src)
        crypto_emb = self.crypto_embedding(crypto_id).unsqueeze(1)
        src = src + crypto_emb

        # Pass through Transformer
        memory = self.transformer_encoder(src)
        # Global average pooling over the sequence dimension
        features = torch.mean(memory, dim=1)

        # Compute logits
        percent_logits = self.percent_change_head(features)
        leg_logits = self.leg_direction_head(features)
        return percent_logits, leg_logits

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class CryptoDataset(Dataset):
    def __init__(self, dataframe, feature_cols, window_size=60):
        self.data = dataframe
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.cryptos = sorted(dataframe['crypto'].unique())
        self.crypto_to_id = {crypto: idx for idx, crypto in enumerate(self.cryptos)}
        self.crypto_data = {
            crypto: dataframe[dataframe['crypto'] == crypto].sort_values('t').reset_index(drop=True)
            for crypto in self.cryptos
        }

        self.indices = []
        for crypto in self.cryptos:
            data_length = len(self.crypto_data[crypto])
            if data_length > self.window_size:
                self.indices.extend([(crypto, idx) for idx in range(data_length - self.window_size)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        crypto, seq_start = self.indices[idx]
        data = self.crypto_data[crypto].iloc[seq_start:seq_start + self.window_size]
        features = data[self.feature_cols].values

        target_idx = seq_start + self.window_size
        data_length = len(self.crypto_data[crypto])
        if target_idx >= data_length:
            target_idx = data_length - 1

        percent_change = self.crypto_data[crypto].iloc[target_idx]['percent_change_classification']
        leg_direction = self.crypto_data[crypto].iloc[target_idx]['leg_direction']

        crypto_id = self.crypto_to_id[crypto]

        return (
            (torch.tensor(features, dtype=torch.float32), torch.tensor(crypto_id, dtype=torch.long)),
            (
                torch.tensor(percent_change, dtype=torch.long),
                torch.tensor(leg_direction, dtype=torch.long),
            ),
        )

def train(model, dataloader, criterion_dict, optimizer, scheduler, scaler, device, accumulation_steps=2):
    model.train()
    epoch_losses = {"percent_change": 0, "leg_direction": 0, "total": 0}
    metrics = {"percent_change_acc": 0, "leg_direction_acc": 0}
    total = 0

    optimizer.zero_grad(set_to_none=True)

    for idx, ((inputs, crypto_ids), (percent_targets, leg_targets)) in enumerate(tqdm(dataloader, desc="Training")):
        inputs = inputs.to(device)
        crypto_ids = crypto_ids.to(device)
        percent_targets = percent_targets.to(device)
        leg_targets = leg_targets.to(device)
        
        with torch.cuda.amp.autocast():
            percent_out, leg_out = model(inputs, crypto_ids)

            loss_percent = criterion_dict['percent_ce'](percent_out, percent_targets)
            loss_leg = criterion_dict['leg_ce'](leg_out, leg_targets)
            
            total_loss = (0.6 * loss_percent + 0.4 * loss_leg) / accumulation_steps
        
        scaler.scale(total_loss).backward()

        batch_size = inputs.size(0)
        total += batch_size
        
        epoch_losses["percent_change"] += loss_percent.item() * batch_size
        epoch_losses["leg_direction"] += loss_leg.item() * batch_size
        epoch_losses["total"] += (total_loss.item() * batch_size * accumulation_steps)
        
        _, predicted_percent = torch.max(percent_out, 1)
        _, predicted_leg = torch.max(leg_out, 1)
        metrics["percent_change_acc"] += (predicted_percent == percent_targets).sum().item()
        metrics["leg_direction_acc"] += (predicted_leg == leg_targets).sum().item()
        
        if (idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
    if (idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    for key in epoch_losses:
        epoch_losses[key] /= total

    metrics["percent_change_acc"] /= total
    metrics["leg_direction_acc"] /= total

    return epoch_losses, metrics

def validate(model, dataloader, criterion_dict, device):
    model.eval()
    val_losses = {"percent_change": 0, "leg_direction": 0, "total": 0}
    val_metrics = {"percent_change_acc": 0, "leg_direction_acc": 0}
    total = 0

    with torch.no_grad():
        all_percent_preds = []
        all_percent_tgts = []
        all_leg_preds = []
        all_leg_tgts = []
        
        for (inputs, crypto_ids), (percent_targets, leg_targets) in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            crypto_ids = crypto_ids.to(device)
            percent_targets = percent_targets.to(device)
            leg_targets = leg_targets.to(device)

            percent_out, leg_out = model(inputs, crypto_ids)

            loss_percent = criterion_dict['percent_ce'](percent_out, percent_targets)
            loss_leg = criterion_dict['leg_ce'](leg_out, leg_targets)
            total_loss = 0.6 * loss_percent + 0.4 * loss_leg

            batch_size = inputs.size(0)
            total += batch_size

            val_losses["percent_change"] += loss_percent.item() * batch_size
            val_losses["leg_direction"] += loss_leg.item() * batch_size
            val_losses["total"] += total_loss.item() * batch_size

            _, predicted_percent = torch.max(percent_out, 1)
            _, predicted_leg = torch.max(leg_out, 1)
            val_metrics["percent_change_acc"] += (predicted_percent == percent_targets).sum().item()
            val_metrics["leg_direction_acc"] += (predicted_leg == leg_targets).sum().item()

            # For optional balanced accuracy
            all_percent_preds.extend(predicted_percent.cpu().numpy())
            all_percent_tgts.extend(percent_targets.cpu().numpy())

    for key in val_losses:
        val_losses[key] /= total

    val_metrics["percent_change_acc"] /= total
    val_metrics["leg_direction_acc"] /= total

    # Example: Compute balanced accuracy for percent_change_classification (optional)
    # ba_percent = balanced_accuracy_score(all_percent_tgts, all_percent_preds)
    # print("Balanced Accuracy (Percent Change):", ba_percent)

    return val_losses, val_metrics

def compute_class_weights(df, target_col):
    class_counts = df[target_col].value_counts().sort_index()
    total = class_counts.sum()
    weights = [total / (len(class_counts) * c) for c in class_counts]
    return torch.tensor(weights, dtype=torch.float)

def main():
    # Parameters
    batch_size = 48
    epochs = 1
    learning_rate = 5e-4
    window_size = 80
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessed_data_dir = 'preprocessed_data'
    dfs = []
    for ticker in os.listdir(preprocessed_data_dir):
        ticker_dir = os.path.join(preprocessed_data_dir, ticker)
        if os.path.isdir(ticker_dir):
            for file in os.listdir(ticker_dir):
                if file.endswith('_preprocessed.csv'):
                    filepath = os.path.join(ticker_dir, file)
                    df = pd.read_csv(filepath)
                    df['crypto'] = ticker
                    dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    print("::full_df:")
    print(full_df)

    target_cols = ['percent_change_classification', 'leg_direction']
    feature_cols = [col for col in full_df.columns if col not in target_cols + ['t', 'crypto']]

    # Drop NaN
    full_df.dropna(subset=feature_cols + target_cols, inplace=True)

    # Normalize features
    scaler = StandardScaler()
    full_df[feature_cols] = scaler.fit_transform(full_df[feature_cols])
    joblib.dump(scaler, 'scaler.joblib')  # Save the scaler
    # Avoid shuffling before splitting
    full_df = full_df.sort_values('t').reset_index(drop=True)  # Ensure data is sorted by time

    train_size = int(0.7 * len(full_df))
    val_size = int(0.15 * len(full_df))

    train_df = full_df.iloc[:train_size]
    val_df = full_df.iloc[train_size:train_size+val_size]
    test_df = full_df.iloc[train_size+val_size:]

    # Print class distributions
    print("Percent change class distribution:")
    print(train_df['percent_change_classification'].value_counts())
    print("Leg direction class distribution:")
    print(train_df['leg_direction'].value_counts())

    # Compute weights for focal loss alpha
    percent_weights = compute_class_weights(train_df, 'percent_change_classification').to(device)
    leg_weights = compute_class_weights(train_df, 'leg_direction').to(device)

    train_dataset = CryptoDataset(train_df, feature_cols, window_size)
    val_dataset = CryptoDataset(val_df, feature_cols, window_size)
    test_dataset = CryptoDataset(test_df, feature_cols, window_size)

    # Prepare WeightedRandomSampler
    # Get targets for weighted sampler
    targets = []
    for i in range(len(train_dataset)):
        _, (pct, _) = train_dataset[i]
        targets.append(int(pct))
    targets = np.array(targets)
    class_sample_counts = np.bincount(targets)
    weight_for_each_class = 1.0 / class_sample_counts
    samples_weight = torch.from_numpy(weight_for_each_class[targets]).double()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    num_features = len(feature_cols)
    num_cryptos = full_df['crypto'].nunique()
    num_classes = full_df['percent_change_classification'].nunique()

    model = ShortTermTransformerModel(
        num_features=num_features,
        num_cryptos=num_cryptos,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=128,
        num_classes=num_classes,
        max_seq_length=window_size
    ).to(device)

    # Using FocalLoss
    percent_focal_loss = FocalLoss(alpha=percent_weights, gamma=1.0)
    leg_focal_loss = FocalLoss(alpha=leg_weights, gamma=1.0)

    criterion_dict = {
        'percent_ce': percent_focal_loss,
        'leg_ce': leg_focal_loss
    }

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 15

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_losses, train_metrics = train(model, train_loader, criterion_dict, optimizer, scheduler, scaler, device, accumulation_steps=2)
        print(f"Training - Losses: {train_losses}")
        print(f"Training - Metrics: {train_metrics}")

        val_losses, val_metrics = validate(model, val_loader, criterion_dict, device)
        print(f"Validation - Losses: {val_losses}")
        print(f"Validation - Metrics: {val_metrics}")

        # Update scheduler based on validation total loss
        scheduler.step(val_losses["total"])

        # Early stopping
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter > early_stopping_patience:
                print("Early stopping triggered.")
                break

        gc.collect()

    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, 'best_short_term_transformer_model.pth')
        print("Best model saved as best_short_term_transformer_model.pth")
    else:
        print("No improvement during training, model not saved.")

    # Final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_losses, test_metrics = validate(model, test_loader, criterion_dict, device)
    print("\nFinal Test Results:")
    print(f"Test Losses: {test_losses}")
    print(f"Test Metrics: {test_metrics}")

def preprocess_and_predict():
    """
    Fetch latest data, preprocess it, and use the model to make predictions.
    Stores predictions separately without altering the original full_df.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Step 1: Fetch latest data
    print("Fetching latest data...")
    collect("2024-09-01", "2024-11-02")

    # Step 2: Preprocess the data
    print("Preprocessing the data...")
    preprocess()

    # Step 3: Load the trained model
    print("Loading the model...")

    preprocessed_data_dir = 'preprocessed_data'
    dfs = []
    for ticker in os.listdir(preprocessed_data_dir):
        ticker_dir = os.path.join(preprocessed_data_dir, ticker)
        if os.path.isdir(ticker_dir):
            for file in os.listdir(ticker_dir):
                if file.endswith('_preprocessed.csv'):
                    filepath = os.path.join(ticker_dir, file)
                    df = pd.read_csv(filepath)
                    df['crypto'] = ticker
                    dfs.append(df)

    if not dfs:
        print("No preprocessed data found. Exiting prediction.")
        return { "status": "failure", "message": "No preprocessed data available for prediction." }

    full_df = pd.concat(dfs, ignore_index=True)
    print("::full_df:")
    print(full_df.head())

    target_cols = ['percent_change_classification', 'leg_direction']
    feature_cols = [col for col in full_df.columns if col not in target_cols + ['t', 'crypto']]

    # Drop NaN
    full_df.dropna(subset=feature_cols + target_cols, inplace=True)

    full_df = full_df.sort_values('t').reset_index(drop=True)  # Ensure data is sorted by time

    # Load the model
    num_features = len(feature_cols)
    num_cryptos = full_df['crypto'].nunique()
    num_classes = full_df['percent_change_classification'].nunique()

    model = ShortTermTransformerModel(
        num_features=num_features,
        num_cryptos=num_cryptos,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=128,
        num_classes=num_classes,
        max_seq_length=80  # Assuming window_size=80
    ).to(device)

    try:
        # Specify map_location to ensure compatibility with the device
        model.load_state_dict(torch.load('best_short_term_transformer_model.pth', map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return { "status": "failure", "message": "Model loading failed." }

    # Step 4: Prepare data for prediction
    print("Preparing data for prediction...")
    # Ensure 'crypto' and 't' columns are present
    if 'crypto' not in full_df.columns or 't' not in full_df.columns:
        print("Error: 'crypto' and/or 't' columns are missing from the preprocessed data.")
        return { "status": "failure", "message": "'crypto' and/or 't' columns are missing from the preprocessed data." }

    scaler = joblib.load('scaler.joblib')  # Load the scaler
    full_df[feature_cols] = scaler.transform(full_df[feature_cols])

    # Initialize an empty list to collect all predictions
    predictions_list = []

    # Iterate over each unique cryptocurrency
    for crypto in tqdm(full_df['crypto'].unique(), desc="Processing Cryptocurrencies"):
        # Filter data for the current crypto
        crypto_df = full_df[full_df['crypto'] == crypto].reset_index(drop=True)

        # Initialize the dataset and dataloader for the current crypto
        crypto_dataset = CryptoDataset(crypto_df, feature_cols, window_size=80)
        crypto_loader = DataLoader(crypto_dataset, batch_size=48, shuffle=False, num_workers=2)

        # Initialize lists to store predictions and corresponding timestamps
        percent_change_predictions = []
        leg_direction_predictions = []
        prediction_timestamps = []

        with torch.no_grad():
            for (inputs, crypto_ids), _ in tqdm(crypto_loader, desc=f"Predicting for {crypto}", leave=False):
                inputs = inputs.to(device)
                crypto_ids = crypto_ids.to(device)
                percent_probs, leg_probs = model(inputs, crypto_ids)

                # Convert logits to predictions
                percent_pred = torch.argmax(percent_probs, dim=1).cpu().tolist()
                leg_pred = torch.argmax(leg_probs, dim=1).cpu().tolist()

                percent_change_predictions.extend(percent_pred)
                leg_direction_predictions.extend(leg_pred)

                # Calculate corresponding timestamps
                # Each prediction corresponds to the row immediately after the window
                start_index = len(percent_change_predictions) - len(percent_pred)
                end_index = start_index + len(percent_pred)
                corresponding_timestamps = crypto_df['t'].iloc[start_index + 80:end_index + 80].tolist()

                prediction_timestamps.extend(corresponding_timestamps)

        # Create a DataFrame for the current crypto's predictions
        crypto_predictions_df = pd.DataFrame({
            't': prediction_timestamps,
            'crypto': [crypto] * len(percent_change_predictions),
            'percent_change_prediction': percent_change_predictions,
            'leg_direction_prediction': leg_direction_predictions
        })

        # Append to the main predictions list
        predictions_list.append(crypto_predictions_df)

    # Concatenate all predictions into a single DataFrame
    predictions_df = pd.concat(predictions_list, ignore_index=True)

    # Save predictions separately
    os.makedirs('predictions', exist_ok=True)
    predictions_df.to_csv('predictions/latest_predictions.csv', index=False)
    print("Predictions saved to 'predictions/latest_predictions.csv'.")
    predictions_dict = predictions_df.to_dict(orient="records")
    return { "status": "success", "predictions": predictions_dict }

def handler(job):
    # Access the input data from the job
    job_input = job.get("input", {})
    
    # Retrieve the 'START_DATE' and 'END_DATE' values
    START_DATE = job_input.get("START_DATE", "")
    print(f"START_DATE :  {START_DATE}")
    END_DATE = job_input.get("END_DATE", "")
    print(f"END_DATE :  {END_DATE}")

    
    # Implement your processing logic using 'start_date' and 'end_date'

    collect(START_DATE, END_DATE)
    preprocess()
    main()
    prediction_result = preprocess_and_predict()
    return prediction_result

runpod.serverless.start({"handler": handler})
