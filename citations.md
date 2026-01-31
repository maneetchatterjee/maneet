# Citations and References

## Core Algorithms

### Soft Actor-Critic (SAC)

**Paper**: "Soft Actor-Critic Algorithms and Applications"

**Authors**: Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, Sergey Levine

**Conference**: ICML 2019

**URL**: https://arxiv.org/abs/1812.05905

**Implementation**: Original implementation adapted from the paper and Spinning Up in Deep RL.

**License**: MIT (research use)

### Dreamer / DreamerV3

**Paper**: "Mastering Diverse Domains through World Models"

**Authors**: Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap

**Conference**: NeurIPS 2023

**URL**: https://arxiv.org/abs/2301.04104

**Implementation**: Simplified version inspired by the paper. Core RSSM dynamics and imagination rollout concepts adapted from DreamerV3.

**Original Implementation**: https://github.com/danijar/dreamerv3

**License**: MIT (research use)

**Related Paper - DreamerV2**: "Mastering Atari with Discrete World Models" (2021)
https://arxiv.org/abs/2010.02193

### DrQv2 (Data Regularized Q)

**Paper**: "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning"

**Authors**: Denis Yarats, Rob Fergus, Alessandro Lazaric, Lerrel Pinto

**Conference**: ICLR 2022

**URL**: https://arxiv.org/abs/2107.09645

**Implementation**: Data augmentation techniques inspired by DrQv2 for image-based policies (optional component).

**License**: MIT (research use)

## Supporting Techniques

### DeepMimic

**Paper**: "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills"

**Authors**: Xue Bin Peng, Pieter Abbeel, Sergey Levine, Michiel van de Panne

**Conference**: SIGGRAPH 2018

**URL**: https://arxiv.org/abs/1804.02717

**Usage**: Imitation distillation concepts for hierarchical controller (optional component).

### Domain Randomization

**Paper**: "Sim-to-Real: Learning Agile Locomotion For Quadruped Robots"

**Authors**: Jie Tan, Tingnan Zhang, Erwin Coumans, Atil Iscen, Yunfei Bai, Danijar Hafner, Steven Bohez, Vincent Vanhoucke

**Conference**: RSS 2018

**URL**: https://arxiv.org/abs/1804.10332

**Usage**: Domain randomization techniques for sim-to-real transfer.

### Hierarchical RL / Options Framework

**Paper**: "Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning"

**Authors**: Richard S. Sutton, Doina Precup, Satinder Singh

**Journal**: Artificial Intelligence, 1999

**URL**: https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf

**Usage**: Options framework concepts for hierarchical controller.

## Software Libraries

### PyBullet

**Library**: PyBullet Physics Simulation

**URL**: https://pybullet.org/

**GitHub**: https://github.com/bulletphysics/bullet3

**License**: Zlib License

**Citation**: Erwin Coumans and Yunfei Bai. PyBullet, a Python module for physics simulation for games, robotics and machine learning. http://pybullet.org, 2016-2021.

### PyTorch

**Library**: PyTorch

**URL**: https://pytorch.org/

**GitHub**: https://github.com/pytorch/pytorch

**License**: BSD 3-Clause

**Citation**: Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. NeurIPS.

### Stable-Baselines3

**Library**: Stable-Baselines3

**URL**: https://stable-baselines3.readthedocs.io/

**GitHub**: https://github.com/DLR-RM/stable-baselines3

**License**: MIT

**Usage**: Reference implementation for SAC baseline comparison (optional).

## Research Use

All implementations in this repository are for **research and educational purposes only**. 

If you use this code in your research, please cite:

1. The original papers for the algorithms you use (listed above)
2. The relevant software libraries
3. This repository (if publishing results)

## Additional Resources

### Learning Resources

- **Spinning Up in Deep RL** (OpenAI): https://spinningup.openai.com/
- **Berkeley Deep RL Course**: https://rail.eecs.berkeley.edu/deeprlcourse/
- **David Silver's RL Course**: https://www.davidsilver.uk/teaching/

### Related Implementations

- **rlkit**: https://github.com/rail-berkeley/rlkit
- **cleanrl**: https://github.com/vwxyzjn/cleanrl
- **tianshou**: https://github.com/thu-ml/tianshou

## Contact

For questions about citations or implementation details:
- Open a GitHub issue
- Include paper title and specific question
- We'll respond with clarification or additional references

## License Compliance

This project complies with the licenses of all dependencies:
- Research use is permitted under MIT and BSD licenses
- Commercial use may require additional licensing
- See individual library licenses for details

**Disclaimer**: This implementation is an educational reimplementation and simplification of the cited works. For production use, consult original implementations and papers.
