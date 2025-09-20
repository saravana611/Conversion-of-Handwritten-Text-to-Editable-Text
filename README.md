# Conversion-of-Handwritten-Text-to-Editable-Text

This project is based on Microsft Trocr model. For Fine tuning with Bentham dataset and adding calibration see the folder named Trocr_bentham_finetuned_calibration.


- `Trocr_bentham_finetuned_calibration/`  
  Contains a fine-tuned TrOCR (Transformer-based OCR) model trained on the Bentham handwriting dataset with calibration for improved confidence and accuracy.



To test the project follow the steps

**1.**  Clone the project.

```
https://github.com/saravana611/Conversion-of-Handwritten-Text-to-Editable-Text.git
```
**2.** Navigate to the project directory.

```
cd Conversion-of-Handwritten-Text-to-Editable-Text
```
**3.** Install the dependencies.

```
pip install -r requirements.txt
```
**4.** Download the Microsoft TROCR trocr-base-handwritten model from [Hugging face models](https://huggingface.co/models?search=microsoft/trocr)

Note you can skip if using your own fine-tuned model.

*  open terminal and run the below line 

```
 python get_models.py
```
**5.** Open terminal and run the demo ocr file

```
python demo_ocr.py
```

**6.** Navigate to the web Address

```
 http://127.0.0.1:7860
```
**7.** That's it you can test your handwritten lines like this

![Demo](sample_demo.png)


## Thank you
