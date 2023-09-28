import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class QLearningTrader:
    def __init__(self, dataset, num_actions, learning_rate, discount_factor, exploration_rate, num_episodes):
        self.dataset = dataset
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.num_episodes = num_episodes
        self.q_table = None

    def train(self):
        self.q_table = np.zeros((len(self.dataset), self.num_actions))

        for _ in range(self.num_episodes):
            state = self._get_initial_state()
            done = False

            while not done:
                action = self._select_action(state)
                next_state, reward, done = self._take_action(state, action)
                self._update_q_table(state, action, reward, next_state)
                state = next_state

    def _get_initial_state(self):
        return 0

    def _select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state])

    def _take_action(self, state, action):
        next_state = state + 1
        reward = self._calculate_reward(state, action)
        done = (next_state == len(self.dataset) - 1)
        return next_state, reward, done

    def _calculate_reward(self, state, action):
        reward = 0.0  # Default reward

        current_price = self.dataset[state, 1]  # Close price
        next_price = self.dataset[state + 1, 1]  # Close price

        if action == 1:  # Buy
            reward = next_price - current_price
        elif action == 2:  # Sell
            reward = current_price - next_price
        elif action == 3:  # Close Buy and Open Sell
            reward = -np.abs(current_price - next_price)
        elif action == 4:  # Close Buy and Open Buy
            reward = 0.0
        elif action == 5:  # Close Sell and Open Sell
            reward = -np.abs(current_price - next_price)
        elif action == 6:  # Close Sell and Open Buy
            reward = 0.0

        return reward

    def _update_q_table(self, state, action, reward, next_state):
        q_value = self.q_table[state, action]
        max_q_value = np.max(self.q_table[next_state])
        new_q_value = q_value + self.learning_rate * (reward + self.discount_factor * max_q_value - q_value)
        self.q_table[state, action] = new_q_value

    def select_best_action(self, state):
        return np.argmax(self.q_table[state])

    def evaluate_model(self):
        cumulative_rewards = []
        cumulative_profits = []
        current_state = self._get_initial_state()
        done = False
        cumulative_reward = 0.0
        cumulative_profit = 0.0

        while not done:
            action = self.select_best_action(current_state)
            next_state, reward, done = self._take_action(current_state, action)
            cumulative_reward += reward
            cumulative_profit += reward
            current_state = next_state

            cumulative_rewards.append(cumulative_reward)
            cumulative_profits.append(cumulative_profit)

        return cumulative_rewards, cumulative_profits

    def calculate_sharpe_ratio(self, returns):
        num_trading_days = len(returns)
        average_daily_return = np.mean(returns)
        daily_std_dev = np.std(returns)
        annualized_return = average_daily_return * num_trading_days
        annualized_std_dev = daily_std_dev * np.sqrt(num_trading_days)
        sharpe_ratio = annualized_return / annualized_std_dev if annualized_std_dev != 0 else 0.0
        return sharpe_ratio

    def calculate_max_drawdown(self, profits):
        max_drawdown = 0.0
        peak = profits[0]

        for profit in profits:
            if profit > peak:
                peak = profit
            drawdown = (peak - profit) / peak if peak != 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown


# Load and preprocess the financial dataset
dataset = pd.read_csv("/content/GBPUSD1440.csv")
dataset = dataset[["Date", "Close"]].values

# Reverse the dataset
reversed_dataset = np.flip(dataset, axis=0)

# Create the Q-learning trader with the reversed dataset
trader = QLearningTrader(reversed_dataset, num_actions=6, learning_rate=0.1, discount_factor=0.9,
                         exploration_rate=0.2, num_episodes=1000)

# Train the Q-learning trader
trader.train()

# Evaluate the model
cumulative_rewards, cumulative_profits = trader.evaluate_model()
sharpe_ratio = trader.calculate_sharpe_ratio(cumulative_profits)
max_drawdown = trader.calculate_max_drawdown(cumulative_profits)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(cumulative_rewards)
plt.xlabel("Day")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Over Time")

plt.subplot(1, 2, 2)
plt.plot(cumulative_profits)
plt.xlabel("Day")
plt.ylabel("Cumulative Profit")
plt.title("Cumulative Profit Over Time")

# Get the last date from the dataset
last_date = pd.to_datetime(dataset[-1, 0])

# Get the next 3 dates for predictions
next_dates = pd.date_range(last_date, periods=120, freq="D")

# Print the next 3 days' predictions
print("Next 13 Days' Predictions:")
for i, date in enumerate(next_dates):
    action = trader.select_best_action(len(dataset) - 120 + i)
    if action == 1:
        action_label = "Buy"
    elif action == 2:
        action_label = "Sell"
    elif action == 3:
        action_label = "Close Buy and Open Sell"
    elif action == 4:
        action_label = "Close Buy and Open Buy"
    elif action == 5:
        action_label = "Close Sell and Open Sell"
    elif action == 6:
        action_label = "Close Sell and Open Buy"
    else:
        action_label = f"Unknown action: {action}"
    print(f"{date.date()}: {action_label}")


plt.tight_layout()
plt.show()

# Print evaluation metrics
print("Sharpe Ratio:", sharpe_ratio)
print("Max Drawdown:", max_drawdown)
