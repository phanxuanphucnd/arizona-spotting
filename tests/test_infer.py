from datetime import datetime
from arizona_spotting.models import Wav2KWS
from arizona_spotting.learners import Wav2KWSLearner

def test_inference():

    model = Wav2KWS(
        num_classes=2,
        model_type='binary',
        encoder_hidden_dim=768,
        out_channels=112,
        pretrained_model='wav2vec-base-en'
    )

    learner = Wav2KWSLearner(model=model)
    learner.load_model(model_path='models/wav2kws_model.pt')

    now = datetime.now()

    print('---------', learner.model_type)

    output = learner.inference(input='data/gsc_v2.0/test/active/0cb74144_nohash_0.wav')
    
    print(output)

    print(f"\nInference time: {(datetime.now() - now)}")
    
test_inference()

# @ray.remote
# def remote_inference(input, learner):
#     result = learner.inference(input)
    
#     return result

# def test_inference_with_ray():

#     ray.shutdown()
#     ray.init(num_cpus=8)

#     model = Wav2KWS(
#         num_classes=2,
#         encoder_hidden_dim=768,
#         out_channels=112,
#         pretrained_model='wav2vec-base-en'
#     )

#     learner = Wav2KWSLearner(model=model)
#     learner.load_model(model_path='./models/wav2kws_model.pt')

#     now = datetime.now()

#     pred = remote_inference.remote(input='data/gsc_v2.0/test/non_active/0c540988_nohash_2.wav', learner=learner)

#     prediction = ray.get(pred)

#     print(f"Time: {datetime.now() - now}")

# test_inference_with_ray()