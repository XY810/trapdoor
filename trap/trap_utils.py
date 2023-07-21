{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/XY810/trapdoor/blob/main/trap/trap_utils.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nmN4-xZ7FZdW",
        "outputId": "19bbb883-00e4-4f01-97c8-c8081a8141b9"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Mounted at /content/drive\n",
            "Wed Jul 19 18:00:08 2023       \n",
            "+-----------------------------------------------------------------------------+\n",
            "| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |\n",
            "|-------------------------------+----------------------+----------------------+\n",
            "| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |\n",
            "| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |\n",
            "|                               |                      |               MIG M. |\n",
            "|===============================+======================+======================|\n",
            "|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |\n",
            "| N/A   45C    P8    10W /  70W |      0MiB / 15360MiB |      0%      Default |\n",
            "|                               |                      |                  N/A |\n",
            "+-------------------------------+----------------------+----------------------+\n",
            "                                                                               \n",
            "+-----------------------------------------------------------------------------+\n",
            "| Processes:                                                                  |\n",
            "|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |\n",
            "|        ID   ID                                                   Usage      |\n",
            "|=============================================================================|\n",
            "|  No running processes found                                                 |\n",
            "+-----------------------------------------------------------------------------+\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# 查看分配到的GPU\n",
        "gpu_info = !nvidia-smi\n",
        "gpu_info = '\\n'.join(gpu_info)\n",
        "if gpu_info.find('failed') >= 0:\n",
        "  print('Not connected to a GPU')\n",
        "else:\n",
        "  print(gpu_info)\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gZ1SaACuVepi",
        "outputId": "3ff32817-33c5-4ec6-e594-60fbd5096158"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "fatal: not a git repository (or any of the parent directories): .git\n"
          ]
        }
      ],
      "source": [
        "!git fetch https://github.com/XY810/trapdoor.git"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HCPrhyyhWzIw",
        "outputId": "892e3280-6d65-4159-ef11-efe2f2621cca"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\u001b[31mERROR: Could not find a version that satisfies the requirement attacks (from versions: none)\u001b[0m\u001b[31m\n",
            "\u001b[0m\u001b[31mERROR: No matching distribution found for attacks\u001b[0m\u001b[31m\n",
            "\u001b[0m"
          ]
        }
      ],
      "source": [
        "!pip install attacks"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RPgZ4uhBIAg3",
        "outputId": "0c770f86-27c7-4491-9750-054ed40c6313"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2023-07-21 09:20:08.905008: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.\n",
            "To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
            "2023-07-21 09:20:09.931830: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT\n",
            "WARNING:tensorflow:From /content/trapdoor/trap/trap_utils.py:23: is_gpu_available (from tensorflow.python.framework.test_util) is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Use `tf.config.list_physical_devices('GPU')` instead.\n",
            "2023-07-21 09:20:13.072026: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "2023-07-21 09:20:13.678690: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "2023-07-21 09:20:13.678996: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "2023-07-21 09:20:16.403776: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "2023-07-21 09:20:16.404127: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "2023-07-21 09:20:16.404361: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "2023-07-21 09:20:16.404511: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:47] Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.\n",
            "2023-07-21 09:20:16.404566: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /device:GPU:0 with 13664 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5\n",
            "2023-07-21 09:20:16.413611: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "2023-07-21 09:20:16.413879: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "2023-07-21 09:20:16.414068: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "2023-07-21 09:20:16.414283: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "2023-07-21 09:20:16.414504: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "2023-07-21 09:20:16.414674: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1635] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13664 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5\n",
            "2023-07-21 09:20:16.415739: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:353] MLIR V1 optimization pass is not enabled\n",
            "Injection Ratio:  0.5\n",
            "Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz\n",
            "11490434/11490434 [==============================] - 0s 0us/step\n",
            "Learning rate:  0.001\n",
            "/usr/local/lib/python3.10/dist-packages/keras/optimizers/legacy/adam.py:117: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.\n",
            "  super().__init__(name, **kwargs)\n",
            "First Step: Training Normal Model...\n",
            "/content/./trapdoor/trap/inject_trapdoor.py:130: UserWarning: `model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.\n",
            "  new_model.fit_generator(clean_train_gen, validation_data=test_nor_gen, steps_per_epoch=number_images // 32,\n",
            "2023-07-21 09:20:17.481107: W tensorflow/c/c_api.cc:300] Operation '{name:'count_1/Assign' id:189 op device:{requested: '', assigned: ''} def:{{{node count_1/Assign}} = AssignVariableOp[_has_manual_control_dependencies=true, dtype=DT_FLOAT, validate_shape=false](count_1, count_1/Initializer/zeros)}}' was changed by setting attribute after it was run by a session. This mutation will have no effect, and will trigger an error in the future. Either don't modify nodes after running them or create a new session.\n",
            "2023-07-21 09:20:17.560180: W tensorflow/c/c_api.cc:300] Operation '{name:'learning_rate/Assign' id:293 op device:{requested: '', assigned: ''} def:{{{node learning_rate/Assign}} = AssignVariableOp[_has_manual_control_dependencies=true, dtype=DT_FLOAT, validate_shape=false](learning_rate, learning_rate/Initializer/initial_value)}}' was changed by setting attribute after it was run by a session. This mutation will have no effect, and will trigger an error in the future. Either don't modify nodes after running them or create a new session.\n",
            "Learning rate:  0.001\n",
            "Epoch 1/10\n",
            "2023-07-21 09:20:17.747602: W tensorflow/c/c_api.cc:300] Operation '{name:'loss_1/mul' id:255 op device:{requested: '', assigned: ''} def:{{{node loss_1/mul}} = Mul[T=DT_FLOAT, _has_manual_control_dependencies=true](loss_1/mul/x, loss_1/dense_loss/value)}}' was changed by setting attribute after it was run by a session. This mutation will have no effect, and will trigger an error in the future. Either don't modify nodes after running them or create a new session.\n",
            "2023-07-21 09:20:17.770616: W tensorflow/c/c_api.cc:300] Operation '{name:'training/Adam/dense1/kernel/v/Assign' id:452 op device:{requested: '', assigned: ''} def:{{{node training/Adam/dense1/kernel/v/Assign}} = AssignVariableOp[_has_manual_control_dependencies=true, dtype=DT_FLOAT, validate_shape=false](training/Adam/dense1/kernel/v, training/Adam/dense1/kernel/v/Initializer/zeros)}}' was changed by setting attribute after it was run by a session. This mutation will have no effect, and will trigger an error in the future. Either don't modify nodes after running them or create a new session.\n",
            "2023-07-21 09:20:21.585680: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:424] Loaded cuDNN version 8900\n",
            "/usr/local/lib/python3.10/dist-packages/keras/engine/training_v1.py:2335: UserWarning: `Model.state_updates` will be removed in a future version. This property should not be used in TensorFlow 2.0, as `updates` are applied automatically.\n",
            "  updates = self.state_updates\n",
            "2023-07-21 09:20:34.053458: W tensorflow/c/c_api.cc:300] Operation '{name:'loss_1/mul' id:255 op device:{requested: '', assigned: ''} def:{{{node loss_1/mul}} = Mul[T=DT_FLOAT, _has_manual_control_dependencies=true](loss_1/mul/x, loss_1/dense_loss/value)}}' was changed by setting attribute after it was run by a session. This mutation will have no effect, and will trigger an error in the future. Either don't modify nodes after running them or create a new session.\n",
            "/content/trapdoor/trap/trap_utils.py:204: UserWarning: `Model.evaluate_generator` is deprecated and will be removed in a future version. Please use `Model.evaluate`, which supports generators.\n",
            "  _, clean_acc = self.model.evaluate_generator(self.test_nor_gen, verbose=0, steps=100)\n",
            "/content/trapdoor/trap/trap_utils.py:205: UserWarning: `Model.evaluate_generator` is deprecated and will be removed in a future version. Please use `Model.evaluate`, which supports generators.\n",
            "  _, attack_acc = self.model.evaluate_generator(self.adv_gen, steps=100, verbose=0)\n",
            "Epoch: 0 - Clean Acc 0.9881 - Trapdoor Acc 0.0950\n",
            "1875/1875 - 18s - loss: 0.1252 - accuracy: 0.9612 - val_loss: 0.0401 - val_accuracy: 0.9881 - lr: 0.0010 - 18s/epoch - 9ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 2/10\n",
            "Epoch: 1 - Clean Acc 0.9903 - Trapdoor Acc 0.1002\n",
            "1875/1875 - 9s - loss: 0.0428 - accuracy: 0.9865 - val_loss: 0.0275 - val_accuracy: 0.9903 - lr: 0.0010 - 9s/epoch - 5ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 3/10\n",
            "Epoch: 2 - Clean Acc 0.9900 - Trapdoor Acc 0.0950\n",
            "1875/1875 - 9s - loss: 0.0301 - accuracy: 0.9902 - val_loss: 0.0298 - val_accuracy: 0.9900 - lr: 0.0010 - 9s/epoch - 5ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 4/10\n",
            "Epoch: 3 - Clean Acc 0.9915 - Trapdoor Acc 0.0988\n",
            "1875/1875 - 9s - loss: 0.0226 - accuracy: 0.9930 - val_loss: 0.0285 - val_accuracy: 0.9915 - lr: 0.0010 - 9s/epoch - 5ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 5/10\n",
            "Epoch: 4 - Clean Acc 0.9916 - Trapdoor Acc 0.1018\n",
            "1875/1875 - 8s - loss: 0.0166 - accuracy: 0.9943 - val_loss: 0.0261 - val_accuracy: 0.9916 - lr: 0.0010 - 8s/epoch - 4ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 6/10\n",
            "Epoch: 5 - Clean Acc 0.9919 - Trapdoor Acc 0.1009\n",
            "1875/1875 - 9s - loss: 0.0137 - accuracy: 0.9954 - val_loss: 0.0274 - val_accuracy: 0.9919 - lr: 0.0010 - 9s/epoch - 5ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 7/10\n",
            "Epoch: 6 - Clean Acc 0.9909 - Trapdoor Acc 0.1063\n",
            "1875/1875 - 9s - loss: 0.0111 - accuracy: 0.9967 - val_loss: 0.0362 - val_accuracy: 0.9909 - lr: 0.0010 - 9s/epoch - 5ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 8/10\n",
            "Epoch: 7 - Clean Acc 0.9931 - Trapdoor Acc 0.0914\n",
            "1875/1875 - 8s - loss: 0.0091 - accuracy: 0.9971 - val_loss: 0.0276 - val_accuracy: 0.9931 - lr: 0.0010 - 8s/epoch - 4ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 9/10\n",
            "Epoch: 8 - Clean Acc 0.9903 - Trapdoor Acc 0.0966\n",
            "1875/1875 - 9s - loss: 0.0101 - accuracy: 0.9966 - val_loss: 0.0408 - val_accuracy: 0.9903 - lr: 0.0010 - 9s/epoch - 5ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 10/10\n",
            "Epoch: 9 - Clean Acc 0.9865 - Trapdoor Acc 0.1000\n",
            "1875/1875 - 9s - loss: 0.0069 - accuracy: 0.9978 - val_loss: 0.0515 - val_accuracy: 0.9865 - lr: 3.1623e-04 - 9s/epoch - 5ms/step\n",
            "Second Step: Injecting Trapdoor...\n",
            "/content/./trapdoor/trap/inject_trapdoor.py:136: UserWarning: `model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.\n",
            "  new_model.fit_generator(trap_train_gen, validation_data=test_nor_gen, steps_per_epoch=number_images // 32,\n",
            "Learning rate:  0.001\n",
            "Epoch 1/10\n",
            "Epoch: 0 - Clean Acc 0.8419 - Trapdoor Acc 0.6840\n",
            "1875/1875 - 12s - loss: 1.6184 - accuracy: 0.5420 - val_loss: 0.5966 - val_accuracy: 0.8419 - lr: 0.0010 - 12s/epoch - 6ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 2/10\n",
            "Epoch: 1 - Clean Acc 0.9500 - Trapdoor Acc 0.9812\n",
            "1875/1875 - 12s - loss: 0.2348 - accuracy: 0.9254 - val_loss: 0.1623 - val_accuracy: 0.9500 - lr: 0.0010 - 12s/epoch - 7ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 3/10\n",
            "Epoch: 2 - Clean Acc 0.9664 - Trapdoor Acc 0.9941\n",
            "1875/1875 - 13s - loss: 0.0795 - accuracy: 0.9754 - val_loss: 0.1041 - val_accuracy: 0.9664 - lr: 0.0010 - 13s/epoch - 7ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 4/10\n",
            "Epoch: 3 - Clean Acc 0.9766 - Trapdoor Acc 0.9969\n",
            "1875/1875 - 13s - loss: 0.0517 - accuracy: 0.9842 - val_loss: 0.0744 - val_accuracy: 0.9766 - lr: 0.0010 - 13s/epoch - 7ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 5/10\n",
            "Epoch: 4 - Clean Acc 0.9753 - Trapdoor Acc 0.9972\n",
            "1875/1875 - 12s - loss: 0.0382 - accuracy: 0.9879 - val_loss: 0.0811 - val_accuracy: 0.9753 - lr: 0.0010 - 12s/epoch - 6ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 6/10\n",
            "Epoch: 5 - Clean Acc 0.9837 - Trapdoor Acc 0.9981\n",
            "1875/1875 - 12s - loss: 0.0317 - accuracy: 0.9903 - val_loss: 0.0620 - val_accuracy: 0.9837 - lr: 0.0010 - 12s/epoch - 7ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 7/10\n",
            "Epoch: 6 - Clean Acc 0.9778 - Trapdoor Acc 0.9978\n",
            "1875/1875 - 13s - loss: 0.0277 - accuracy: 0.9909 - val_loss: 0.0656 - val_accuracy: 0.9778 - lr: 0.0010 - 13s/epoch - 7ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 8/10\n",
            "Epoch: 7 - Clean Acc 0.9844 - Trapdoor Acc 0.9984\n",
            "1875/1875 - 12s - loss: 0.0220 - accuracy: 0.9931 - val_loss: 0.0533 - val_accuracy: 0.9844 - lr: 0.0010 - 12s/epoch - 7ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 9/10\n",
            "Epoch: 8 - Clean Acc 0.9846 - Trapdoor Acc 0.9997\n",
            "1875/1875 - 12s - loss: 0.0196 - accuracy: 0.9936 - val_loss: 0.0548 - val_accuracy: 0.9846 - lr: 0.0010 - 12s/epoch - 7ms/step\n",
            "Learning rate:  0.001\n",
            "Epoch 10/10\n",
            "Epoch: 9 - Clean Acc 0.9816 - Trapdoor Acc 0.9991\n",
            "1875/1875 - 12s - loss: 0.0180 - accuracy: 0.9940 - val_loss: 0.0535 - val_accuracy: 0.9816 - lr: 0.0010 - 12s/epoch - 6ms/step\n",
            "2023-07-21 09:23:59.945229: W tensorflow/c/c_api.cc:300] Operation '{name:'dense_1/bias/Assign' id:629 op device:{requested: '', assigned: ''} def:{{{node dense_1/bias/Assign}} = AssignVariableOp[_has_manual_control_dependencies=true, dtype=DT_FLOAT, validate_shape=false](dense_1/bias, dense_1/bias/Initializer/zeros)}}' was changed by setting attribute after it was run by a session. This mutation will have no effect, and will trigger an error in the future. Either don't modify nodes after running them or create a new session.\n",
            "2023-07-21 09:24:00.176133: W tensorflow/c/c_api.cc:300] Operation '{name:'dense1_1/bias/v/Assign' id:848 op device:{requested: '', assigned: ''} def:{{{node dense1_1/bias/v/Assign}} = AssignVariableOp[_has_manual_control_dependencies=true, dtype=DT_FLOAT, validate_shape=false](dense1_1/bias/v, dense1_1/bias/v/Initializer/zeros)}}' was changed by setting attribute after it was run by a session. This mutation will have no effect, and will trigger an error in the future. Either don't modify nodes after running them or create a new session.\n",
            "/content/./trapdoor/trap/inject_trapdoor.py:145: UserWarning: `Model.evaluate_generator` is deprecated and will be removed in a future version. Please use `Model.evaluate`, which supports generators.\n",
            "  loss, acc = new_model.evaluate_generator(test_nor_gen, verbose=0, steps=100)\n",
            "2023-07-21 09:24:00.352751: W tensorflow/c/c_api.cc:300] Operation '{name:'loss_2/mul' id:744 op device:{requested: '', assigned: ''} def:{{{node loss_2/mul}} = Mul[T=DT_FLOAT, _has_manual_control_dependencies=true](loss_2/mul/x, loss_2/dense_loss/value)}}' was changed by setting attribute after it was run by a session. This mutation will have no effect, and will trigger an error in the future. Either don't modify nodes after running them or create a new session.\n",
            "/content/./trapdoor/trap/inject_trapdoor.py:148: UserWarning: `Model.evaluate_generator` is deprecated and will be removed in a future version. Please use `Model.evaluate`, which supports generators.\n",
            "  loss, backdoor_acc = new_model.evaluate_generator(test_adv_gen, steps=200, verbose=0)\n",
            "File saved to trapdoor/results/mnist_res.p, use this path as protected-path for the eval script. \n"
          ]
        }
      ],
      "source": [
        "!python3 ./trapdoor/trap/inject_trapdoor.py"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!$git commit"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FwY-rynXxOET",
        "outputId": "c343613c-1fd2-4ba9-af35-a68a7ffec40c"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/bin/bash: commit: command not found\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cwhDPzJhpj8F"
      },
      "outputs": [],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "m7EI-VWyVarK"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyM0TGuBO8lG+HGtS3XMO0OJ",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}