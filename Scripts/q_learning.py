
"""Tabular Q-learning agent for footRein with persistent metrics."""

import json
import os
import random

ACTIONS = ("nearest", "forward", "open")


class QLearningAgent:

    def __init__(
        self,
        path=None,
        alpha=0.15,
        gamma=0.90,
        epsilon=0.15
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.path = path or os.path.join(
            os.path.dirname(__file__),
            "q_table.json"
        )

        self.q = {}
        self.steps = 0

        self.history = {
            "steps": [],
            "rewards": [],
            "td_errors": [],
            "td_losses": [],
            "mean_q": [],
            "mean_q_by_action": {
                a: [] for a in ACTIONS
            },
            "action_counts": {
                a: 0 for a in ACTIONS
            },
            "visited_states": []
        }

        self._load()

    @staticmethod
    def encode_state(x, y, pressure, goal_distance):
        x_zone = min(5, max(0, int(x // 200)))

        if y < 240:
            y_lane = 0
        elif y < 460:
            y_lane = 1
        else:
            y_lane = 2

        if pressure < 45:
            pressure_zone = 0
        elif pressure < 100:
            pressure_zone = 1
        else:
            pressure_zone = 2

        goal_zone = min(
            5, max(0, int(goal_distance // 200))
        )

        return (
            f"{x_zone}:{y_lane}:"
            f"{pressure_zone}:{goal_zone}"
        )

    def _values(self, state, actions):
        row = self.q.setdefault(state, {})

        for action in actions:
            row.setdefault(action, 0.0)

        return row

    def choose_action(self, state, actions):
        """Choose a tactic using epsilon-greedy selection."""

        row = self._values(state, actions)

        if random.random() < self.epsilon:
            action = random.choice(list(actions))
        else:
            best_value = max(row[a] for a in actions)
            best_actions = [
                a for a in actions
                if row[a] == best_value
            ]
            action = random.choice(best_actions)

        self.history["action_counts"][action] = (
            self.history["action_counts"].get(action, 0) + 1
        )

        return action

    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        actions,
        terminal=False
    ):
        """Perform one Q-learning update and record diagnostics."""

        reward = float(reward)

        row = self._values(state, actions)
        old_value = row[action]

        if terminal:
            target = reward
        else:
            next_row = self._values(next_state, actions)
            target = reward + self.gamma * max(
                next_row[a] for a in actions
            )

        td_error = target - old_value
        row[action] = old_value + self.alpha * td_error

        self.steps += 1
        h = self.history

        h["steps"].append(self.steps)
        h["rewards"].append(reward)
        h["td_errors"].append(float(td_error))
        h["td_losses"].append(float(td_error ** 2))

        all_values = [
            self.q[s].get(a, 0.0)
            for s in self.q
            for a in actions
        ]

        h["mean_q"].append(
            sum(all_values) / len(all_values)
            if all_values else 0.0
        )

        for a in ACTIONS:
            values = [
                self.q[s].get(a, 0.0)
                for s in self.q
            ]

            h["mean_q_by_action"][a].append(
                sum(values) / len(values) if values else 0.0
            )

        h["visited_states"].append(len(self.q))

        # Save every update so reports can use the latest data.
        self._save()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as file:
                data = json.load(file)

            self.q = data.get("q", {})
            self.steps = int(data.get("steps", 0))

            saved = data.get("history", {})

            for key in (
                "steps",
                "rewards",
                "td_errors",
                "td_losses",
                "mean_q",
                "visited_states"
            ):
                if isinstance(saved.get(key), list):
                    self.history[key] = saved[key]

            saved_counts = saved.get("action_counts", {})
            for action in ACTIONS:
                self.history["action_counts"][action] = int(
                    saved_counts.get(action, 0)
                )

            saved_means = saved.get("mean_q_by_action", {})
            for action in ACTIONS:
                values = saved_means.get(action, [])
                if isinstance(values, list):
                    self.history["mean_q_by_action"][action] = values

        except (OSError, ValueError, TypeError):
            # A missing or invalid file starts a fresh table.
            self.q = {}
            self.steps = 0

    def _save(self):
        try:
            folder = os.path.dirname(self.path) or "."
            os.makedirs(folder, exist_ok=True)

            temp_path = self.path + ".tmp"

            with open(temp_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "q": self.q,
                        "steps": self.steps,
                        "history": self.history
                    },
                    file,
                    indent=2
                )

            os.replace(temp_path, self.path)

        except OSError as error:
            print("Could not save Q-table:", error)
