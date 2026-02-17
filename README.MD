# NextHAM: Advancing Universal Deep Learning for Electronic-Structure Hamiltonian Prediction of Materials

This is the official implementation of the paper **"Advancing Universal Deep Learning for Electronic-Structure Hamiltonian Prediction of Materials"**, accepted at **ICLR 2026**.

---

## ⚙️ Requirements

### Hardware
Both training and inference were conducted on high-performance hardware:
* **GPU**: 4x NVIDIA A800 (80GiB VRAM each).
* *Note: Due to the high dimensionality of Hamiltonian matrices, a large GPU memory is recommended for optimal performance.*

### Software Environment
The software dependencies are managed via Conda. All required packages are specified in the `./environment.yml` file.

To set up the environment, run:
```bash
# Create the environment from the file
conda env create -f environment.yml

# Activate the environment
conda activate nextham
```

## 📊 Dataset Preparation

The **Materials-SOC-HAM** dataset proposed in this paper is publicly available.

1. **Download**: You can access the dataset via the link below:
   * **Link**: [Materials-SOC-HAM Dataset](https://dzefile.hpccube.com:65011/efile/s/w/bmV4dGhhbQ==_c3c50c552df97ace&)
   * **Extraction Code**: `DGEo`

2. **Configuration**: 
   Once the dataset is downloaded and extracted to your local directory, you must update the file roots:
   * Locate the following files: `datasets/train.txt`, `datasets/val.txt`, and `datasets/test.txt`.
   * Replace the placeholder `/your_path/` in each file with the absolute path to your local dataset directory.

---

## 🚀 Training and Evaluation

1. **Pre-trained Models** (Optional)

    If you wish to train on new data, we highly recommend fine-tuning from our provided pre-trained models to achieve faster convergence.

    * **Download Link**: [NextHAM Pre-trained Weights](https://dzefile.hpccube.com:65011/efile/s/w/bmV4dGhhbQ==_191f3754a48697e8&)
    * **Password**: `QoYA`
    * **Setup**: Please download the weights and place them in the ./pretrained_models directory.

2. **Training**
    To start the model training and validation process, execute the following script:
    ```bash
    sh scripts/train/train_val.sh
    ```

3. **Testing**
    To test the model after training, execute the following script:
    ```bash
    sh scripts/test/test.sh
    ```

## 📝 Citations
    @inproceedings{yin2026nextham,
    title={Advancing Universal Deep Learning for Electronic-Structure Hamiltonian Prediction of Materials},
    author={Shi Yin, Zujian Dai, Xinyang Pan, and Lixin He},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2026}
    }