## Quick Start
1. Setup AndroidWorld Environment
   1. Download Android Studio [here](https://developer.android.com/studio?gad_source=1&gclid=Cj0KCQjw3ZayBhDRARIsAPWzx8oLcadBD0vAq8xmUutaunLGSzhgEtLz4xVZ_SpV4G0xJazS7LxQkDsaAuveEALw_wcB&gclsrc=aw.ds)
   2. Create an Android Virtual Device (AVD) by following these instructions. For hardware select **Pixel 6**, for System Image select **Tiramisu, API Level 33**, and choose AVD name as **AndroidWorldAvd**. [Watch the setup video.](https://github.com/google-research/android_world/assets/162379927/efc33980-8b36-44be-bb2b-a92d4c334a50)

2. Launch the Android Emulator from the command line
    Launch the emulator from the command line, not using the Android Studio UI, with the `-grpc 8554` flag which is needed communication with accessibility forwarding app.

    ```bash
    # Typically it's located in ~/Android/Sdk/emulator/emulator or
    # ~/Library/Android/sdk/emulator/emulator
    EMULATOR_NAME=AndroidWorldAvd # From previous step
    ~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554
    ```

3. [Optional] It's recommended to use `conda`, which you can download [here](https://docs.anaconda.com/free/miniconda/miniconda-install/).

    ```
    conda create -n android_world python=3.11.8
    conda activate android_world
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install -y numpy pandas
    ```

4. Install Dependency. *Note: Python 3.11 or above is required.*

    ```python
    pip install -r requirements.txt
    ```

5. Modify vLLM.
     
    Please navigate to vllm/model_executor/layers/sampler.py, add the following to line 317.

    ```python
    for val, lst in zip(logits, sample_logprobs):
            for d in lst:
                for k in d.keys():
                    d[k].logprob = val
    ```
    (See https://github.com/vllm-project/vllm/issues/11397 for more explanations)

6. Add model provider APIs as environment variables.

    Three API providers are supported: OpenAI and its compatible APIs, and Azure OpenAI services. You may configure any of these based on your preferences.
   
   **These APIs are only used for building the working memory, V-Droid allows to build the working memory without using these third-party APIs**
    ```bash
    # Add to .bashrc.

    # use Gemini GCP service, which requires API key
    export GCP_API_KEY=
    
    # use openai compatible APIs, including OPENAI, Qwen and DeepSeek
    export OPENAI_ENDPOINT=
    export OPENAI_MODEL_NAME=
    export OPENAI_API_VERSION=
    export OPENAI_API_KEY=


    # use azure openai services
    export AZURE_OPENAI_API_KEY=
    export AZURE_OPENAI_MODEL_NAME=
    export AZURE_OPENAI_API_VERSION=
    export AZURE_OPENAI_ENDPOINT=
    ```

8. Download Lora weights for V-Droid model
   
   The V-Droid model weight is available at https://huggingface.co/V-Droid/V-Droid-8B-0323


9. Lauanch the emulator and run the eveluation tasks
   ```bash
   emulator -avd AndroidWorldAvd -no-window -no-snapshot -grpc 8554
   bash main.sh
   ```


10. Training 
   You may use the following code to train the lora module in V-Droid. We provide several training pairs to use.
   ```bash
   train.sh 
   ```
