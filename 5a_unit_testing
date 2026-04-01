import json
import unittest
from unittest.mock import patch

from EmotionDetection import emotion_detector


class MockResponse:
    def __init__(self, payload):
        self.text = json.dumps(payload)


def mocked_post(url, headers=None, json=None):
    text = json["raw_document"]["text"]

    responses = {
        "I am glad this happened": {
            "emotionPredictions": [
                {
                    "emotion": {
                        "anger": 0.02,
                        "disgust": 0.03,
                        "fear": 0.04,
                        "joy": 0.88,
                        "sadness": 0.03,
                    }
                }
            ]
        },
        "I am really mad about this": {
            "emotionPredictions": [
                {
                    "emotion": {
                        "anger": 0.86,
                        "disgust": 0.04,
                        "fear": 0.03,
                        "joy": 0.02,
                        "sadness": 0.05,
                    }
                }
            ]
        },
        "I feel disgusted just hearing about this": {
            "emotionPredictions": [
                {
                    "emotion": {
                        "anger": 0.06,
                        "disgust": 0.81,
                        "fear": 0.03,
                        "joy": 0.02,
                        "sadness": 0.08,
                    }
                }
            ]
        },
        "I am so sad about this": {
            "emotionPredictions": [
                {
                    "emotion": {
                        "anger": 0.03,
                        "disgust": 0.02,
                        "fear": 0.04,
                        "joy": 0.01,
                        "sadness": 0.90,
                    }
                }
            ]
        },
        "I am really afraid that this will happen": {
            "emotionPredictions": [
                {
                    "emotion": {
                        "anger": 0.04,
                        "disgust": 0.03,
                        "fear": 0.87,
                        "joy": 0.01,
                        "sadness": 0.05,
                    }
                }
            ]
        },
    }

    return MockResponse(responses[text])


class TestEmotionDetection(unittest.TestCase):
    @patch("final_project.emotion_detection.requests.post", side_effect=mocked_post)
    def test_emotion_detector(self, mock_post):
        test_cases = [
            ("I am glad this happened", "joy"),
            ("I am really mad about this", "anger"),
            ("I feel disgusted just hearing about this", "disgust"),
            ("I am so sad about this", "sadness"),
            ("I am really afraid that this will happen", "fear"),
        ]

        for statement, expected_emotion in test_cases:
            with self.subTest(statement=statement):
                result = emotion_detector(statement)
                self.assertEqual(result["dominant_emotion"], expected_emotion)


if __name__ == "__main__":
    unittest.main()
