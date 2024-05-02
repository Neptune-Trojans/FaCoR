from typing import Optional
import ast
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from face_alignment import align
from final_statistics import baseline_model2
from utils import get_device


class PredictionsCreator:
    def __init__(self, threshold: float, model_path: str, device: str, images_root: str):
        self._threshold = threshold
        self._model = baseline_model2(model_path, device)
        self._device = device
        self._images_root = images_root

    def _prepare_image(self, image_path):

        image = align.get_aligned_face(image_path)
        if image is None:
            return None
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        image = image[np.newaxis, :]
        image = torch.from_numpy(image).type(torch.float).to(device=self._device)
        return image

    def prediction(self, image1_path: str, image2_path: str) -> Optional[float]:
        image1 = self._prepare_image(image1_path)
        image2 = self._prepare_image(image2_path)

        if image1 is None or image2 is None:
            return None

        em1, em2, x1, x2, _ = self._model([image1, image2])
        pred = torch.cosine_similarity(em1, em2, dim=1).cpu().detach().numpy().tolist()
        pred = pred[0]
        return pred

    def create_predictions(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        unit_predictions = []
        related_unrelated_scores = []
        similarity_scores = []

        for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
            img1 = row['images1'][0]
            img2 = row['images2'][0]
            img1 = os.path.join(self._images_root, img1)
            img2 = os.path.join(self._images_root, img2)

            prediction = self.prediction(img1, img2)

            if prediction is None:
                # detector issues
                unit_predictions.append('NOT_ENOUGH_VISUAL_INFORMATION')
                related_unrelated_scores.append([0.0, 0.0])
                similarity_scores.append(0.0)
                continue

            result = 'UNRELATED'
            if prediction > self._threshold:
                result = 'RELATED'

            unit_predictions.append(result)
            related_unrelated_scores.append([prediction, prediction])
            similarity_scores.append(prediction)

        pairs_df['unit_predictions'] = unit_predictions
        pairs_df['related_unrelated_scores'] = related_unrelated_scores
        pairs_df['similarity_scores'] = similarity_scores
        pairs_df['scores'] = [s[0] for s in related_unrelated_scores]

        return pairs_df


if __name__ == '__main__':
    threshold = 0.10982190817594528
    model_path = '/Users/yudkin/dev/FaCoR/output_data/model.pth'
    device = get_device()
    pairs_data = '/Users/yudkin/Documents/Datasets/facebook_test2/verify_all.csv'
    images_root = '/Users/yudkin/Documents/Datasets/facebook_test2'
    pairs_df = pd.read_csv(pairs_data)

    pairs_df['images1'] = pairs_df['images1'].apply(ast.literal_eval)
    pairs_df['images2'] = pairs_df['images2'].apply(ast.literal_eval)
    pairs_df = pairs_df[pairs_df.source == 'facebook']

    p = PredictionsCreator(threshold, model_path, device, images_root)
    pairs_df = p.create_predictions(pairs_df)
    pairs_df.to_csv('pairs_unit_predictions.csv')