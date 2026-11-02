# Submitted model

`round-2.joblib` is the selected classifier head fitted on YAMNet embeddings. It is the original frozen artifact, not a retrained or converted model. Run `python3 scripts/setup_models.py` to verify it and copy it into the path expected by `live-config.json`.

- Classifier SHA-256: `d3547ad8493ba740e3d575848a39703ee7a05247bdff2d99d51514f3102bd14a`
- Bundled encoder: `../test-audio/model/yamnet.tflite`
- Encoder SHA-256: `10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de`

YAMNet is Google's pretrained AudioSet sound classifier. Its model card identifies Apache 2.0 licensing: https://www.kaggle.com/models/google/yamnet/tfLite/classification-tflite/1
The Apache 2.0 license is in `../third_party/YAMNet-LICENSE`. Model code and attribution: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet

Joblib files use Python serialization; load only the bundled, verified artifact or other checkpoints you trust.
