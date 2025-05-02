# GNR602_project

# Image Segmentation Application

This application allows you to perform image segmentation on grayscale images using clustering techniques. It provides both a web-based interface using Streamlit and a command-line alternative.

## 📦 Requirements

Ensure that the following files are present in the same directory:

- `app.py`
- `script.py`

## 🔧 Installation

Install the required Python libraries (if not already installed):

```bash
pip install streamlit pillow numpy scipy joblib tqdm
```

## 🚀 Steps to Run the Application

### Option 1: Using the Web Interface

1. Open your terminal.
2. Run the application with:

   ```bash
   streamlit run app.py
   ```

3. Once started, it will automatically open in your default browser at:

   ```
   http://localhost:8501
   ```

4. Upload a **grayscale image** and choose the number of clusters (`k`).
5. The segmented output will be displayed on the screen.

### Option 2: Using the Command Line

1. Run the script directly:

   ```bash
   python3.10 script.py --k X
   ```

   Replace `X` with the desired number of clusters (default is `4`).

2. This will:
   - Process **all grayscale images** in the `images/` folder.
   - Save the segmented outputs to the `output/` folder.

