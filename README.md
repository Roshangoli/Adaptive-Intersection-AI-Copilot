# 🚦 Adaptive Intersection AI Copilot

An AI-powered traffic light controller that optimizes for **both vehicles and pedestrians** using Reinforcement Learning, designed for the Grainger Computing Innovation Prize.

## 🎯 Project Overview

This project addresses the critical issue of pedestrian safety at intersections while maintaining efficient vehicle flow. Traditional traffic lights are timed for averages or vehicle-first flow, often leaving pedestrians—especially vulnerable groups like students, children, seniors, and people with disabilities—waiting too long and encouraging risky jaywalking.

### Key Innovation
- **Dual Optimization**: Balances pedestrian wait times with vehicle throughput
- **Real-time Adaptation**: Uses live counts and short-term forecasts
- **Explainable AI**: Provides natural language explanations for decisions
- **Privacy-Focused**: Blurs faces/license plates, stores only anonymized metrics
- **Safety Constraints**: Prevents risky jaywalking while maintaining vehicle flow

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Computer      │    │   Reinforcement │    │   Forecasting   │
│   Vision        │    │   Learning       │    │   & Data        │
│   (YOLO +       │    │   (PPO/DQN)      │    │   (Prophet/     │
│   Tracking)     │    │                  │    │   LSTM)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   AI Controller │
                    │   (Explainable) │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   SUMO Traffic   │
                    │   Simulation     │
                    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- macOS/Linux (tested on macOS 23.5.0)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Adaptive-Intersection-AI-Copilot
   ```

2. **Set up the environment**
   ```bash
   make setup
   ```

3. **Run a simulation**
   ```bash
   # Fixed-time controller
   make sim-fixed
   
   # RL-based controller
   make sim-rl
   ```

4. **Launch the dashboard**
   ```bash
   make dashboard
   ```

## 📁 Project Structure

```
Adaptive-Intersection-AI-Copilot/
├── sim/                    # SUMO simulation files
│   ├── 4_way_intersection.net.xml  # Traffic network
│   ├── routes.rou.xml              # Vehicle routes
│   ├── pedestrians.rou.xml         # Pedestrian routes
│   ├── sim.sumocfg                # SUMO configuration
│   ├── run_fixed.py               # Fixed-time controller
│   └── run_rl.py                  # RL-based controller
├── dashboard/              # Streamlit web dashboard
│   └── app.py             # Dashboard application
├── src/                   # Source code modules
│   ├── cv/                # Computer Vision
│   ├── data/              # Data processing & forecasting
│   ├── rl/                # Reinforcement Learning
│   └── explainability/    # AI explainability
├── results/               # Simulation outputs
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── requirements.txt      # Python dependencies
├── Makefile              # Build automation
└── README.md             # This file
```

## 🎮 Available Commands

| Command | Description |
|---------|-------------|
| `make setup` | Set up virtual environment and install dependencies |
| `make sim-fixed` | Run fixed-time traffic controller simulation |
| `make sim-rl` | Run RL-based traffic controller simulation |
| `make dashboard` | Launch Streamlit dashboard |
| `make test` | Run tests |
| `make clean` | Clean up generated files |
| `make help` | Show all available commands |

## 🔬 Technical Details

### Simulation Environment
- **SUMO**: Eclipse SUMO 1.21.0+ for traffic simulation
- **TraCI**: Python API for real-time traffic control
- **Network**: 4-way intersection with pedestrian crossings

### AI Components
- **RL Agent**: Q-learning with epsilon-greedy exploration
- **State Space**: Vehicle/pedestrian counts per direction
- **Action Space**: Traffic light phases (4 phases)
- **Reward Function**: Balances wait times and throughput

### Computer Vision
- **YOLO**: Object detection for pedestrians/vehicles
- **Tracking**: Multi-object tracking for counting
- **Privacy**: On-device face/license plate blurring

### Forecasting
- **Prophet**: Time series forecasting for demand prediction
- **LSTM**: Deep learning for complex patterns
- **Context**: Weather and event data integration

## 📊 Results & Metrics

The system tracks key performance indicators:

- **Vehicle Wait Time**: Average waiting time for vehicles
- **Pedestrian Wait Time**: Average waiting time for pedestrians
- **Throughput**: Total vehicles and pedestrians processed
- **Safety**: Reduction in near-miss incidents
- **Efficiency**: Overall intersection performance

## 🎯 Competition Goals

This project is designed for the **Grainger Computing Innovation Prize** with focus on:

1. **Social Impact**: Safer, fairer crossings for vulnerable pedestrians
2. **Innovation**: Joint optimization of pedestrians and vehicles
3. **Implementation**: Practical, scalable solution
4. **Explainability**: Transparent AI decisions
5. **Privacy**: Privacy-preserving data collection

## 👥 Team

- **Venkata Sai Roshan** (PM/Cloud) — Project management and cloud deployment
- **Teammate 2** (Simulation/RL) — Reinforcement learning agent development
- **Teammate 3** (Computer Vision) — Pedestrian/vehicle counting systems
- **Teammate 4** (Data/Forecast/Explainability) — Forecasting and explainability

## 🔧 Development

### Adding New Features

1. **Computer Vision**: Add new detection models in `src/cv/`
2. **RL Agent**: Implement new algorithms in `src/rl/`
3. **Forecasting**: Add forecasting models in `src/data/`
4. **Dashboard**: Extend visualization in `dashboard/app.py`

### Testing

```bash
# Run all tests
make test

# Run specific test
python -m pytest tests/test_specific.py -v
```

### Code Quality

```bash
# Format code
black src/ dashboard/ sim/

# Lint code
flake8 src/ dashboard/ sim/
```

## 📚 Documentation

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [TraCI Python API](https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Eclipse SUMO team for the traffic simulation platform
- OpenAI for the GPT models used in development
- The Grainger Computing Innovation Prize organizers
- Our team members and advisors

---

**Status**: 🚧 In Development for Grainger Computing Innovation Prize Round 2  
**Last Updated**: October 2024  
**Version**: 1.0.0