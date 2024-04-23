import requests
import re

import gradio as gr
import numpy as np
import pandas as pd
from torch import topk
from torch.nn.functional import softmax
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers_interpret import ImageClassificationExplainer

## loads class labels for the ImageNet dataset from a URL and returns them as a list.
def load_label_data():
    dosya_url = "https://raw.githubusercontent.com/god-hephaestus/vit-explainer/main/vit-files/ImageNet_1k.txt"
    response = requests.get(dosya_url)
    labels = []
    pattern = '["\'](.*?)["\']'
    for line in response.text.split('\n'):
        try:
            tmp = re.findall(pattern, line)[0]
            labels.append(tmp)
        except IndexError:
            pass
    return labels

## This class manages the UI and model inference logic.
## It initializes the ViT model and processor, loads label data, and defines methods for running the model, classifying images, and explaining predictions.
class WebUI:
    def __init__(self):
        super().__init__()
        self.nb_classes = 10
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.labels = load_label_data()
        self.result = {"A":2}

## run_model() method runs the ViT model on an input image and returns the top k predictions.
    
    def run_model(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        outputs = softmax(outputs.logits, dim=1)
        outputs = topk(outputs, k=self.nb_classes)
        return outputs
    
## classify_image() method classifies an image and returns the top k predicted classes along with their probabilities.
    def classify_image(self, image):
        top10 = self.run_model(image)
        self.result = {self.labels[top10[1][0][i]]: float(top10[0][0][i]) for i in range(self.nb_classes)}
        self.redraw_frame(self)
        return self.result
    
    def redraw_frame(self, image):
        print(self.result)
        keys = list(self.result.keys())
        values = list(self.result.values())
        simple = pd.DataFrame(
        {
        "item": keys,
        "inventory": values,
        }
        )        
        return gr.BarPlot(
            value=simple,
            x="item",
            y="inventory",
            title="Simple Bar Plot",
            container=False
        )


## explain_pred() method generates saliency maps to explain model predictions.

    def explain_pred(self, image):
        image_classification_explainer = ImageClassificationExplainer(model=self.model, feature_extractor=self.processor)
        saliency = image_classification_explainer(image)
        saliency = np.squeeze(np.moveaxis(saliency, 1, 3))
        saliency[saliency >= 0.05] = 0.05
        saliency[saliency <= -0.05] = -0.05
        saliency /= np.amax(np.abs(saliency))
        return saliency
    
## run() method sets up the UI elements using Gradio and launches the UI.

    def run(self):
        with gr.Blocks() as demo:
            gr.HTML(value="<p style='margin-top: 1rem, margin-bottom: 1rem'>This is html</p>")
            with gr.Accordion("Detaylar", open=True):
                gr.Markdown("Böyle Böyle çalışıyor vs vs")
            gr.Markdown("""# ViT Bitirme Projesi
                        Görüntü Sınıflandırmaya başlamak için örnek fotoğraf seçin ya da fotoğraf yükleyin""")
            with gr.Row(elem_classes="customclass",variant="panel",equal_height=True):
                image = gr.Image(height=512)
                print(self.result)
                
                label = self.redraw_frame(image)
                # label = gr.Label(num_top_classes=self.nb_classes)
                saliency = gr.Image(height=512, label="attention (saliency) map", show_label=True)

                with gr.Column(scale=0.2, min_width=150):
                    run_btn = gr.Button("Analysis", variant="primary", elem_id="run-button")

                    run_btn.click(
                        fn=lambda x: self.explain_pred(x),
                        inputs=image,
                        outputs=saliency,
                    )

                    run_btn.click(
                        fn=lambda x: self.classify_image(x),
                        inputs=image,
                        outputs=label,
                    )

                    gr.Examples(
                        examples=[
                            ['https://raw.githubusercontent.com/god-hephaestus/vit-explainer/main/vit-files/car.jpg'],
                            ['https://raw.githubusercontent.com/god-hephaestus/vit-explainer/main/vit-files/hare.jpg'],
                            ['https://raw.githubusercontent.com/god-hephaestus/vit-explainer/main/vit-files/horse.jpg'],
                        ],
                        inputs=image,
                        outputs=image,
                        fn=lambda x: x,
                        cache_examples=False,
                    )
            
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
## creates an instance of the WebUI class and runs the UI.

def main():
    ui = WebUI()
    ui.run()


if __name__ == "__main__":
    main()
