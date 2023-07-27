# 🎉 Tiny Decoder Transformer

## 📚 Summary

Welcome to the Tiny Decoder Transformer repository! This is my personal sandbox for experimenting with various improvements in the transformer architecture. The goal is to learn and improve my understanding by implementing everything from scratch. The `Tiny Stories dataset` serves as the benchmark for these experiments.

## 📚 Dataset

The `Tiny Stories dataset` is a unique collection of short stories synthesized by GPT-3.5 and GPT-4. It uses vocabulary that's typically understood by 3 to 4-year-olds. This feature makes it ideal for my experiments, as it supports training and evaluating smaller, yet efficient Language Models (LMs). More information about this dataset can be found in the [published paper](https://arxiv.org/abs/2305.07759).

## 🎯 Gameplan

Here's the roadmap for the improvements and experiments:

1. **Start with the Base Decoder**: Implement the base decoder from scratch, as done in the [Verne Decoder Transformer](https://github.com/joaoflf/verne-decoder-transformer) project.

2. **Experiment with Sophisticated Positional Encoders**: Develop more advanced positional encoders to enhance sequence position information.

3. **Test Different Activation Functions**: Evaluate the impact of different activation functions, such as Swish and GELU, on the model's performance.

4. **Implement Flash Attention**: Incorporate the Flash Attention mechanism to reduce the computational complexity of transformers. For more details about Flash Attention, refer to this [published paper](<insert_paper_link>).

5. **Explore Layer Normalization Techniques**: Experiment with different layer normalization techniques like Post-LN and Pre-LN.

6. **Try Alternative Attention Mechanisms**: Implement and test attention mechanisms, such as Linformer and LongFormer, aimed at reducing computational complexity.

7. **Apply Pruning Techniques**: Investigate various pruning techniques to create a leaner model without compromising performance.

8. **Experiment with Model Distillation**: Use distillation techniques to develop smaller and more efficient models.

9. **Implement Dynamic Depth**: Introduce dynamic depth to customize the network depth for each input to potentially enhance efficiency.

## 📜 License

This project is licensed under the terms of the MIT license.

For more information, refer to the LICENSE file in this repository.

