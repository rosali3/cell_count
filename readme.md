Here’s a user-friendly README file for the **Cell Counter Telegram Bot**:

---

# Cell Counter Telegram Bot

A powerful yet easy-to-use tool for counting cells in microscopy images. This bot leverages advanced deep-learning models to detect and segment cell cultures, providing accurate cell counts alongside annotated images.

## 📌 [Try it out on Telegram](https://t.me/cell_counter_dstu_bot)

---

## Features

- **Automatic Cell Counting**: Counts objects such as adipocytes, fibroblasts, and mesenchymal stem cells from your uploaded images.
- **Annotated Results**: Sends back an image with a segmentation mask overlay, making it easy to visualize detected cells.
- **Model Flexibility**: Choose from four YOLOv8-based segmentation models trained for different cell cultures.

---

## Supported Cell Cultures

The bot is designed for three main types of cell cultures:
1. **Culture 1**: Adipocytes  
2. **Culture 2**: Fibroblasts  
3. **Culture 3**: Mesenchymal Stem Cells  

All models are fine-tuned using our custom dataset, which you can explore [here](https://app.roboflow.com/cultures/cell-segmentation-ci5n3/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true).

---

## How It Works

1. **Start the Bot**: Click on the [bot link](https://t.me/cell_counter_dstu_bot) and press the **Start** button.
2. **Choose a Model**: Select one of the following models:
   - A model trained on **all three cultures**.
   - Models fine-tuned specifically for one culture (Adipocytes, Fibroblasts, or Mesenchymal Stem Cells).
3. **Upload an Image**: Send a microscopy image of a cell culture.
4. **Receive Results**: The bot will:
   - Count the number of detected objects.
   - Return an image with a segmentation mask applied.
   - Include the cell counts in the reply message.

---

## Models

The bot employs **YOLOv8-segmentation models**, trained as follows:
- **General Model**: Fine-tuned on all three cell cultures.
- **Culture-Specific Models**: Three separate models, each fine-tuned on a specific culture.

### Limitations
- These models are optimized for the provided dataset. Performance on other datasets or images with very different characteristics may vary.

---

## Requirements for Uploaded Images

- **Format**: JPEG, PNG, or BMP.
- **Size**: Ensure images are not excessively large to avoid processing delays.
- **Content**: Microscopy images of the supported cell cultures.

---

## Dataset

The dataset used to fine-tune our models is publicly available. It contains expertly annotated images of the three supported cell cultures.  
👉 [Explore the dataset on Roboflow](https://app.roboflow.com/cultures/cell-segmentation-ci5n3/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true).

---

## Limitations and Future Plans

- **Overlap Issues**: Some cells may not be detected in cases of dense clustering or overlapping.
- **Enhancements**: We're continuously improving detection accuracy and reducing loss from overlapping bounding boxes.

---

## Support

If you encounter any issues or have suggestions for improvement, feel free to reach out!
