#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <iostream>

namespace py = pybind11;

// MonteCarloES Class Definition
class MonteCarloES {
public:
    MonteCarloES(py::object env, int num_episodes)
        : env(env), num_episodes(num_episodes), gen(rd()) {}

    std::unordered_map<int, double> run() {
        std::unordered_map<int, double> value_table;
        std::unordered_map<int, std::vector<double>> returns;

        for (int i = 0; i < num_episodes; ++i) {
            env.attr("reset")();
            auto episode = run_episode(env);

            double G = 0;
            for (int t = episode.size() - 1; t >= 0; --t) {
                auto[state, action, reward, next_state] = episode[t];
                G = reward + 0.99 * G;  // Assuming discount factor gamma = 0.99
                bool first_visit = true;
                for (int k = 0; k < t; ++k) {
                    if (std::get<0>(episode[k]) == state) {
                        first_visit = false;
                        break;
                    }
                }
                if (first_visit) {
                    returns[state].push_back(G);
                    value_table[state] = std::accumulate(returns[state].begin(), returns[state].end(), 0.0) / returns[state].size();
                }
            }
        }

        return value_table;
    }

private:
    std::vector<std::tuple<int, int, double, int>> run_episode(py::object env) {
        std::vector<std::tuple<int, int, double, int>> episode;
        int state = env.attr("state_id")().cast<int>();
        while (!env.attr("is_game_over")().cast<bool>()) {
            auto actions = env.attr("available_actions")().cast<std::vector<int>>();
            std::uniform_int_distribution<> dis(0, actions.size() - 1);
            int action = actions[dis(gen)];
            env.attr("step")(action);
            double reward = env.attr("score")().cast<double>();
            int next_state = env.attr("state_id")().cast<int>();
            episode.emplace_back(state, action, reward, next_state);
            state = next_state;
        }
        return episode;
    }

    py::object env;
    int num_episodes;
    std::random_device rd;
    std::mt19937 gen;
};

class OffPolicyMonteCarloControl {
public:
    OffPolicyMonteCarloControl(py::object env, int episodes, float epsilon = 0.1, float discount_factor = 0.99)
        : env(env), episodes(episodes), epsilon(epsilon), discount_factor(discount_factor), gen(rd()) {}

    std::unordered_map<int, std::unordered_map<int, double>> run() {
        std::unordered_map<int, std::unordered_map<int, double>> value_table;
        std::unordered_map<int, std::unordered_map<int, double>> C;

        for (int i = 0; i < episodes; ++i) {
            env.attr("reset")();
            auto episode = run_episode(env);

            double G = 0;
            double W = 1;
            for (int t = episode.size() - 1; t >= 0; --t) {
                auto[state, action, reward, next_state] = episode[t];
                G = reward + discount_factor * G;
                C[state][action] += W;
                value_table[state][action] += (W / C[state][action]) * (G - value_table[state][action]);

                // Get the best action for the current state
                auto best_action = std::distance(value_table[state].begin(), std::max_element(value_table[state].begin(), value_table[state].end(), [](const auto& a, const auto& b) {
                    return a.second < b.second;
                }));

                if (action != best_action) {
                    break;
                }
                W *= 1.0 / (1.0 - epsilon + epsilon / env.attr("num_actions")().cast<int>());
            }
        }

        return value_table;
    }

private:
    std::vector<std::tuple<int, int, double, int>> run_episode(py::object env) {
        std::vector<std::tuple<int, int, double, int>> episode;
        int state = env.attr("state_id")().cast<int>();
        while (!env.attr("is_game_over")().cast<bool>()) {
            auto actions = env.attr("available_actions")().cast<std::vector<int>>();
            std::uniform_int_distribution<> dis(0, actions.size() - 1);
            int action = actions[dis(gen)];
            env.attr("step")(action);
            double reward = env.attr("score")().cast<double>();
            int next_state = env.attr("state_id")().cast<int>();
            episode.emplace_back(state, action, reward, next_state);
            state = next_state;
        }
        return episode;
    }

    py::object env;
    int episodes;
    float epsilon;
    float discount_factor;
    std::random_device rd;
    std::mt19937 gen;
};

class onPolicyMonteCarloControl {
public:
    onPolicyMonteCarloControl(py::object env, int episodes, float epsilon = 0.1, float discount_factor = 0.99)
        : env(env), episodes(episodes), epsilon(epsilon), discount_factor(discount_factor), gen(rd()) {}

    std::unordered_map<int, std::unordered_map<int, double>> run() {
        std::unordered_map<int, std::unordered_map<int, double>> value_table;
        std::unordered_map<int, std::unordered_map<int, std::vector<double>>> returns;

        for (int i = 0; i < episodes; ++i) {
            env.attr("reset")();
            auto episode = run_episode(env);

            double G = 0;
            for (int t = episode.size() - 1; t >= 0; --t) {
                auto [state, action, reward, next_state] = episode[t];
                G = reward + discount_factor * G;
                if (std::find_if(episode.begin(), episode.begin() + t, [state, action](const auto& step) {
                    return std::get<0>(step) == state && std::get<1>(step) == action;
                }) == episode.begin() + t) {
                    returns[state][action].push_back(G);
                    value_table[state][action] = std::accumulate(returns[state][action].begin(), returns[state][action].end(), 0.0) / returns[state][action].size();
                }
            }
        }

        return value_table;
    }

private:
    std::vector<std::tuple<int, int, double, int>> run_episode(py::object env) {
        std::vector<std::tuple<int, int, double, int>> episode;
        int state = env.attr("state_id")().cast<int>();
        std::uniform_real_distribution<> dis(0.0, 1.0);

        while (!env.attr("is_game_over")().cast<bool>()) {
            auto actions = env.attr("available_actions")().cast<std::vector<int>>();
            int action;
            if (dis(gen) < epsilon) {
                std::uniform_int_distribution<> action_dis(0, actions.size() - 1);
                action = actions[action_dis(gen)];
            } else {
                if (value_table.find(state) == value_table.end()) {
                    value_table[state] = std::unordered_map<int, double>();
                }
                for (const auto& act : actions) {
                    if (value_table[state].find(act) == value_table[state].end()) {
                        value_table[state][act] = 0.0;
                    }
                }
                action = actions[std::distance(value_table[state].begin(), std::max_element(value_table[state].begin(), value_table[state].end(), [](const auto& a, const auto& b) {
                    return a.second < b.second;
                }))];
            }
            env.attr("step")(action);
            double reward = env.attr("score")().cast<double>();
            int next_state = env.attr("state_id")().cast<int>();
            episode.emplace_back(state, action, reward, next_state);
            state = next_state;
        }
        return episode;
    }

    py::object env;
    int episodes;
    float epsilon;
    float discount_factor;
    std::random_device rd;
    std::mt19937 gen;
    std::unordered_map<int, std::unordered_map<int, double>> value_table;
};

// QLearning Class Definition
class QLearning {
public:
    QLearning(py::object env, double learning_rate = 0.1, double discount_factor = 0.99, double exploration_rate = 1.0, double exploration_decay = 0.995, double min_exploration_rate = 0.01, int episodes = 10)
        : env(env), learning_rate(learning_rate), discount_factor(discount_factor), exploration_rate(exploration_rate), exploration_decay(exploration_decay), min_exploration_rate(min_exploration_rate), episodes(episodes), gen(rd()) {}

    py::array_t<double> run() {
        int num_states = env.attr("num_states")().cast<int>();
        int num_actions = env.attr("num_actions")().cast<int>();
        py::array_t<double> q_table({num_states, num_actions});
        auto q_table_ptr = q_table.mutable_data();
        
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int episode = 0; episode < episodes; ++episode) {
            env.attr("reset")();
            int state = env.attr("state_id")().cast<int>();
            double total_reward = 0;

            while (!env.attr("is_game_over")().cast<bool>()) {
                int action;
                if (dis(gen) < exploration_rate) {
                    auto actions = env.attr("available_actions")().cast<std::vector<int>>();
                    if (actions.empty()) {
                        std::cerr << "No available actions!" << std::endl;
                        break;
                    }
                    std::uniform_int_distribution<> action_dis(0, actions.size() - 1);
                    action = actions[action_dis(gen)];
                } else {
                    action = std::distance(q_table_ptr + state * num_actions, std::max_element(q_table_ptr + state * num_actions, q_table_ptr + (state + 1) * num_actions));
                }

                env.attr("step")(action);
                int next_state = env.attr("state_id")().cast<int>();
                double reward = env.attr("score")().cast<double>();

                int best_next_action = std::distance(q_table_ptr + next_state * num_actions, std::max_element(q_table_ptr + next_state * num_actions, q_table_ptr + (next_state + 1) * num_actions));
                double td_target = reward + discount_factor * q_table_ptr[next_state * num_actions + best_next_action];
                double td_error = td_target - q_table_ptr[state * num_actions + action];
                q_table_ptr[state * num_actions + action] += learning_rate * td_error;

                state = next_state;
                total_reward += reward;
            }

            exploration_rate = std::max(min_exploration_rate, exploration_rate * exploration_decay);
            std::cout << "Episode " << episode + 1 << ": Total Reward: " << total_reward << ", Exploration Rate: " << exploration_rate << std::endl;
        }

        return q_table;
    }

private:
    py::object env;
    double learning_rate;
    double discount_factor;
    double exploration_rate;
    double exploration_decay;
    double min_exploration_rate;
    int episodes;
    std::random_device rd;
    std::mt19937 gen;
};

// PolicyIteration Class Definition
class PolicyIteration {
public:
    PolicyIteration(py::object env_class, double gamma = 0.99, double theta = 0.95)
        : env_class(env_class), gamma(gamma), theta(theta) {
        env = env_class();
        num_states = env.attr("num_states")().cast<int>();
        num_actions = env.attr("num_actions")().cast<int>();
    }

    std::tuple<std::vector<int>, std::vector<double>> run() {
        // Step 1: Initialize policy randomly
        std::vector<int> policy(num_states);
        std::generate(policy.begin(), policy.end(), [this]() { return rand() % num_actions; });

        // Step 2: Policy Iteration
        std::vector<double> V(num_states, 0.0);

        while (true) {
            policy_evaluation(policy, V);
            bool policy_stable = policy_improvement(policy, V);
            if (policy_stable) {
                break;
            }
        }

        return {policy, V};
    }

private:
    py::object env_class;
    py::object env;
    double gamma;
    double theta;
    int num_states;
    int num_actions;

    void policy_evaluation(std::vector<int>& policy, std::vector<double>& V) {
        while (true) {
            double delta = 0;
            for (int s = 0; s < num_states; ++s) {
                double v = 0;
                for (int a = 0; a < num_actions; ++a) {
                    if (policy[s] == a) {
                        for (int next_state = 0; next_state < num_states; ++next_state) {
                            for (int r_index = 0; r_index < env.attr("num_rewards")().cast<int>(); ++r_index) {
                                double prob = env.attr("p")(s, a, next_state, r_index).cast<double>();
                                double reward = env.attr("reward")(r_index).cast<double>();
                                v += prob * (reward + gamma * V[next_state]);
                            }
                        }
                    }
                }
                delta = std::max(delta, std::abs(v - V[s]));
                V[s] = v;
                std::cout << "Delta: " << delta << std::endl;
            }
            if (delta < theta) {
                break;
            }
        }
    }

    bool policy_improvement(std::vector<int>& policy, std::vector<double>& V) {
        bool policy_stable = true;
        for (int s = 0; s < num_states; ++s) {
            int chosen_a = policy[s];
            auto action_values = one_step_lookahead(s, V);
            int best_a = std::distance(action_values.begin(), std::max_element(action_values.begin(), action_values.end()));
            if (chosen_a != best_a) {
                policy_stable = false;
            }
            policy[s] = best_a;
        }
        return policy_stable;
    }

    std::vector<double> one_step_lookahead(int state, std::vector<double>& V) {
        std::vector<double> A(num_actions, 0.0);
        for (int a = 0; a < num_actions; ++a) {
            for (int next_state = 0; next_state < num_states; ++next_state) {
                for (int r_index = 0; r_index < env.attr("num_rewards")().cast<int>(); ++r_index) {
                    double prob = env.attr("p")(state, a, next_state, r_index).cast<double>();
                    double reward = env.attr("reward")(r_index).cast<double>();
                    A[a] += prob * (reward + gamma * V[next_state]);
                }
            }
        }
        return A;
    }
};

// ValueIteration Class Definition
class ValueIteration {
public:
    ValueIteration(py::object env_class, double gamma = 0.99, double theta = 1e-3)
        : env_class(env_class), gamma(gamma), theta(theta) {
        env = env_class();
        num_states = env.attr("num_states")().cast<int>();
        num_actions = env.attr("num_actions")().cast<int>();
    }

    std::vector<double> run() {
        std::vector<double> V(num_states, 0.0);
        while (true) {
            double delta = 0;
            for (int s = 0; s < num_states; ++s) {
                auto action_values = one_step_lookahead(s, V);
                double max_action_value = *std::max_element(action_values.begin(), action_values.end());
                delta = std::max(delta, std::abs(max_action_value - V[s]));
                V[s] = max_action_value;
            }
            if (delta < theta) {
                break;
            }
        }
        return V;
    }

private:
    py::object env_class;
    py::object env;
    double gamma;
    double theta;
    int num_states;
    int num_actions;

    std::vector<double> one_step_lookahead(int state, const std::vector<double>& V) {
        std::vector<double> A(num_actions, 0.0);
        for (int a = 0; a < num_actions; ++a) {
            for (int next_state = 0; next_state < num_states; ++next_state) {
                for (int r_index = 0; r_index < env.attr("num_rewards")().cast<int>(); ++r_index) {
                    double prob = env.attr("p")(state, a, next_state, r_index).cast<double>();
                    double reward = env.attr("reward")(r_index).cast<double>();
                    A[a] += prob * (reward + gamma * V[next_state]);
                }
            }
        }
        return A;
    }
};

// SARSA Class Definition
class SARSA {
public:
    SARSA(py::object env, double learning_rate = 0.1, double discount_factor = 0.99, double exploration_rate = 1.0, double exploration_decay = 0.995, double min_exploration_rate = 0.01, int episodes = 10)
        : env(env), learning_rate(learning_rate), discount_factor(discount_factor), exploration_rate(exploration_rate), exploration_decay(exploration_decay), min_exploration_rate(min_exploration_rate), episodes(episodes), gen(rd()) {}

    py::array_t<double> run() {
        auto num_states = env.attr("num_states")().cast<int>();
        auto num_actions = env.attr("num_actions")().cast<int>();
        py::array_t<double> q_table({num_states, num_actions});
        auto q_table_ptr = q_table.mutable_data();
        
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int episode = 0; episode < episodes; ++episode) {
            env.attr("reset")();
            int state = env.attr("state_id")().cast<int>();
            auto actions = env.attr("available_actions")().cast<std::vector<int>>();
            std::uniform_int_distribution<> action_dis(0, actions.size() - 1);
            int action = dis(gen) < exploration_rate ? actions[action_dis(gen)] : std::distance(q_table_ptr + state * num_actions, std::max_element(q_table_ptr + state * num_actions, q_table_ptr + (state + 1) * num_actions));

            double total_reward = 0;

            while (!env.attr("is_game_over")().cast<bool>()) {
                env.attr("step")(action);
                int next_state = env.attr("state_id")().cast<int>();
                double reward = env.attr("score")().cast<double>();
                actions = env.attr("available_actions")().cast<std::vector<int>>();
                action_dis = std::uniform_int_distribution<>(0, actions.size() - 1);
                int next_action = dis(gen) < exploration_rate ? actions[action_dis(gen)] : std::distance(q_table_ptr + next_state * num_actions, std::max_element(q_table_ptr + next_state * num_actions, q_table_ptr + (next_state + 1) * num_actions));

                double td_target = reward + discount_factor * q_table_ptr[next_state * num_actions + next_action];
                double td_error = td_target - q_table_ptr[state * num_actions + action];
                q_table_ptr[state * num_actions + action] += learning_rate * td_error;

                state = next_state;
                action = next_action;
                total_reward += reward;
            }

            exploration_rate = std::max(min_exploration_rate, exploration_rate * exploration_decay);
            std::cout << "Episode " << episode + 1 << ": Total Reward: " << total_reward << ", Exploration Rate: " << exploration_rate << std::endl;
        }

        return q_table;
    }

private:
    py::object env;
    double learning_rate;
    double discount_factor;
    double exploration_rate;
    double exploration_decay;
    double min_exploration_rate;
    int episodes;
    std::random_device rd;
    std::mt19937 gen;
};

// DynaQ Class Definition
class DynaQ {
public:
    DynaQ(py::object env, int planning_steps = 5, double learning_rate = 0.1, double discount_factor = 0.99, double exploration_rate = 1.0, double exploration_decay = 0.995, double min_exploration_rate = 0.01, int episodes = 10)
        : env(env), planning_steps(planning_steps), learning_rate(learning_rate), discount_factor(discount_factor), exploration_rate(exploration_rate), exploration_decay(exploration_decay), min_exploration_rate(min_exploration_rate), episodes(episodes), gen(rd()) {}

    py::array_t<double> run() {
        auto num_states = env.attr("num_states")().cast<int>();
        auto num_actions = env.attr("num_actions")().cast<int>();
        py::array_t<double> q_table({num_states, num_actions});
        auto q_table_ptr = q_table.mutable_data();
        
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::unordered_map<int, std::unordered_map<int, std::tuple<int, double>>> model;

        for (int episode = 0; episode < episodes; ++episode) {
            env.attr("reset")();
            int state = env.attr("state_id")().cast<int>();
            double total_reward = 0;

            while (!env.attr("is_game_over")().cast<bool>()) {
                int action;
                if (dis(gen) < exploration_rate) {
                    auto actions = env.attr("available_actions")().cast<std::vector<int>>();
                    std::uniform_int_distribution<> action_dis(0, actions.size() - 1);
                    action = actions[action_dis(gen)];
                } else {
                    action = std::distance(q_table_ptr + state * num_actions, std::max_element(q_table_ptr + state * num_actions, q_table_ptr + (state + 1) * num_actions));
                }

                env.attr("step")(action);
                int next_state = env.attr("state_id")().cast<int>();
                double reward = env.attr("score")().cast<double>();

                int best_next_action = std::distance(q_table_ptr + next_state * num_actions, std::max_element(q_table_ptr + next_state * num_actions, q_table_ptr + (next_state + 1) * num_actions));
                double td_target = reward + discount_factor * q_table_ptr[next_state * num_actions + best_next_action];
                double td_error = td_target - q_table_ptr[state * num_actions + action];
                q_table_ptr[state * num_actions + action] += learning_rate * td_error;

                model[state][action] = std::make_tuple(next_state, reward);

                for (int i = 0; i < planning_steps; ++i) {
                    int simulated_state = rand() % num_states;
                    if (model.find(simulated_state) == model.end()) continue;
                    int simulated_action = rand() % num_actions;
                    if (model[simulated_state].find(simulated_action) == model[simulated_state].end()) continue;
                    auto[simulated_next_state, simulated_reward] = model[simulated_state][simulated_action];
                    int simulated_best_next_action = std::distance(q_table_ptr + simulated_next_state * num_actions, std::max_element(q_table_ptr + simulated_next_state * num_actions, q_table_ptr + (simulated_next_state + 1) * num_actions));
                    double simulated_td_target = simulated_reward + discount_factor * q_table_ptr[simulated_next_state * num_actions + simulated_best_next_action];
                    double simulated_td_error = simulated_td_target - q_table_ptr[simulated_state * num_actions + simulated_action];
                    q_table_ptr[simulated_state * num_actions + simulated_action] += learning_rate * simulated_td_error;
                }

                state = next_state;
                total_reward += reward;
            }

            exploration_rate = std::max(min_exploration_rate, exploration_rate * exploration_decay);
            std::cout << "Episode " << episode + 1 << ": Total Reward: " << total_reward << ", Exploration Rate: " << exploration_rate << std::endl;
        }

        return q_table;
    }

private:
    py::object env;
    int planning_steps;
    double learning_rate;
    double discount_factor;
    double exploration_rate;
    double exploration_decay;
    double min_exploration_rate;
    int episodes;
    std::random_device rd;
    std::mt19937 gen;
};

// Bindings
PYBIND11_MODULE(lib, m) {
    py::class_<MonteCarloES>(m, "MonteCarloES")
        .def(py::init<py::object, int>())
        .def("run", &MonteCarloES::run);

    py::class_<QLearning>(m, "QLearning")
        .def(py::init<py::object, double, double, double, double, double, int>(),
             py::arg("env"),
             py::arg("learning_rate") = 0.1,
             py::arg("discount_factor") = 0.99,
             py::arg("exploration_rate") = 1.0,
             py::arg("exploration_decay") = 0.995,
             py::arg("min_exploration_rate") = 0.01,
             py::arg("episodes") = 10)
        .def("run", &QLearning::run);

    py::class_<PolicyIteration>(m, "PolicyIteration")
        .def(py::init<py::object, double, double>(),
             py::arg("env_class"),
             py::arg("gamma") = 0.99,
             py::arg("theta") = 0.95)
        .def("run", &PolicyIteration::run);

    py::class_<OffPolicyMonteCarloControl>(m, "OffPolicyMonteCarloControl")
        .def(py::init<py::object, int, float, float>(),
             py::arg("env"),
             py::arg("episodes"),
             py::arg("epsilon") = 0.1,
             py::arg("discount_factor") = 0.99)
        .def("run", &OffPolicyMonteCarloControl::run);

    py::class_<onPolicyMonteCarloControl>(m, "onPolicyMonteCarloControl")
        .def(py::init<py::object, int, float, float>(),
             py::arg("env"),
             py::arg("episodes"),
             py::arg("epsilon") = 0.1,
             py::arg("discount_factor") = 0.99)
        .def("run", &onPolicyMonteCarloControl::run);

    py::class_<ValueIteration>(m, "ValueIteration")
        .def(py::init<py::object, double, double>(),
             py::arg("env_class"),
             py::arg("gamma") = 0.99,
             py::arg("theta") = 1e-3)
        .def("run", &ValueIteration::run);

    py::class_<SARSA>(m, "SARSA")
        .def(py::init<py::object, double, double, double, double, double, int>(),
             py::arg("env"),
             py::arg("learning_rate") = 0.1,
             py::arg("discount_factor") = 0.99,
             py::arg("exploration_rate") = 1.0,
             py::arg("exploration_decay") = 0.995,
             py::arg("min_exploration_rate") = 0.01,
             py::arg("episodes") = 10)
        .def("run", &SARSA::run);

    py::class_<DynaQ>(m, "DynaQ")
        .def(py::init<py::object, int, double, double, double, double, double, int>(),
             py::arg("env"),
             py::arg("planning_steps") = 5,
             py::arg("learning_rate") = 0.1,
             py::arg("discount_factor") = 0.99,
             py::arg("exploration_rate") = 1.0,
             py::arg("exploration_decay") = 0.995,
             py::arg("min_exploration_rate") = 0.01,
             py::arg("episodes") = 10)
        .def("run", &DynaQ::run);
}
