# [ICLR 2026] NextHAM: Advancing Universal Deep Learning for Electronic-Structure Hamiltonian Prediction of Materials

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

---

## 📊 Dataset Preparation

The **Materials-SOC-HAM** dataset proposed in this paper is publicly available.

1. **Download**: You can access the dataset via the link below:
   * **Link**: [Materials-SOC-HAM Dataset](https://dzefile.hpccube.com:65011/efile/s/w/bmV4dGhhbQ==_c3c50c552df97ace&)
   * **Extraction Code**: `Ugnh`

2. **Configuration**: 
   Once the dataset is downloaded and extracted to your local directory, you must update the file roots:
   * Locate the following files: `datasets/train.txt`, `datasets/val.txt`, and `datasets/test.txt`.
   * Replace the placeholder `/your_path/` in each file with the absolute path to your local dataset directory.

---

## 🚀 Training and Evaluation

1. **Pre-trained Models** (Optional)

   If you wish to train on new data, we highly recommend fine-tuning from our provided pre-trained models to achieve faster convergence.

   * **Download Link**: [NextHAM Pre-trained Weights]( https://dzefile.hpccube.com:65011/efile/s/w/bmV4dGhhbQ==_191f3754a48697e8& )
   * **Password**: `vORL`
   * **Setup**: Please download the weights and place them in the `./pretrained_models` directory.

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

---

## 🛠️ Custom Dataset Preparation and End-to-End Pipeline

This section provides a step-by-step tutorial on how to prepare your own dataset using ABACUS, train the NextHAM model on it, and visualize the predicted band structures against the Density Functional Theory (DFT) ground truth.

### Step 1: Environment & Dependency Setup
1. Log in to [www.scnet.cn](http://www.scnet.cn) and download `abacus-v3.10.0LTS` and `abacus-v3.9.0.18` from the App Store. Deploy them to your local directory.
2. Update the ABACUS executable paths in `./ABACUS2NextHAM/src/abacus2nextham/constant.py` (lines 12-15).
3. Download the `ase_abacus` package and activate its environment:
   ```bash
   source /your_path/apprepo/ase_abacus/v1.9.16-matplotlib/scripts/env.sh
   ```
   *⚠️ **Important Note**: Sourcing this script initializes a new sub-environment. You will need to reinstall some of the dependencies from `./environment.yml` into this active environment.*

### Step 2: Generate Raw Data via ABACUS
1. Configure your custom working paths in `./ABACUS2NextHAM/src/abacus2nextham/constant.py` (lines 5-10).
2. Place your target ABACUS structure files (`STRU`) into `./ABACUS2NextHAM/input/structures/`. This directory supports batch processing for multiple structures.
3. Navigate to the processing directory and run the calculation:
   ```bash
   cd ./ABACUS2NextHAM
   python src/abacus2nextham/main.py
   ```
4. Wait for the Self-Consistent Field (SCF) calculations to complete. The raw output data will be saved in the `./ABACUS2NextHAM/simulate/` directory.

### Step 3: Convert Data to NextHAM Format
Execute the conversion script to process the raw ABACUS output into NextHAM-compatible equivariant tensors:
```bash
python run_convert.py
```
The processed `.pth` data files will be stored in the `processed_data/` directory. Concurrently, a `data.txt` file will be generated in the same directory, recording the absolute paths of all processed data samples.

### Step 4: Train and Evaluate on Your Custom Dataset
1. Navigate back to the project root directory.
2. Manually split the `data.txt` (generated in Step 3) into three separate files: `train.txt`, `val.txt`, and `test.txt`. Save them under the `./datasets/` directory.
3. Follow the standard training and testing commands detailed in the **Training and Evaluation** section above.
4. **Required for Visualization:** If you intend to plot band structures later, you must save the model's predictions during testing. **Uncomment line 440 in `test.py`** before running the test script. Predictions will be saved in the `./res/` directory by default.

### Step 5: Band Structure Visualization and Comparison
To reconstruct the predicted Hamiltonians into band structures and visually compare them with the DFT ground truth:
```bash
cd ./NextHAM2Bands
sh run_bands.sh
```
*💡 **Tip**: The `run_bands.sh` script uses `case-0` (the first sample of your custom dataset) as a default example. You can modify the script to replace `case-0` with any specific sample name/ID you wish to analyze.*

---

## 📝 Citations

If you find our work and dataset useful in your research, please cite our paper:

```bibtex
@inproceedings{yin2026nextham,
  title={Advancing Universal Deep Learning for Electronic-Structure Hamiltonian Prediction of Materials},
  author={Shi Yin, Zujian Dai, Xinyang Pan, and Lixin He},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
