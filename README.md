# Deep Fake Project

This project contains code for the pipeline of fake video generation. This project is divided into 2 parts.

## Installation

Before starting the installation, first clone the repository and open the terminal at the project root
```shell
git clone https://github.com/beicenter/aiavatar.git
cd aiavatar
```

First of all, we need to create a new virtual environment. To create a python virtual environment, please run the following command in the terminal

```shell
# Create the environment (Recommended: Python==3.8)
python -m venv myenv
# Activate it by running following commands
source venv/bin/activate # For Linux/MacOS
source venv/Scripts/activate # For Windows
```

Now follow the steps to download and install the dependencies and pre-trained models.

1. First download the Real-ESRGAN model by running following command:
    ```shell
    wget 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA' -O 'wav2lip/checkpoints/wav2lip_gan.pth'
    ```
   If you don't have _wget_ support then simply [click here](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA) to download and save the file at following path `wav2lip/checkpoints/wav2lip_gan.pth` under your project root directory.
2. Download Tacotron and Waveglow model from following links and place it under the `checkpoints` repository.

    https://ngc.nvidia.com/catalog/models/nvidia:tacotron2pyt_fp16/files?version=3 > `checkpoints > tacotron2`
    https://ngc.nvidia.com/catalog/models/nvidia:waveglow256pyt_fp16/files?version=2 > `checkpoints > waveglow`

3. Install the dependencies by running following command:

    ```shell
    pip install -r requirements.txt
    ```

## Inference

To do the inference, run the following command:
```shell
python run.py --text "ADD YOUR TEXT HERE" --video <PATH_TO_DRIVING_VIDEO> --output <PATH_TO_OUTPUT_VIDEO>
```
