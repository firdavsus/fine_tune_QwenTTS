The scipts to fine tune Qwen TTS on a new language

step by step:
1. load the model using-> load.py (by default loads 0.6B base model)
2. load dataset using-> format.py (change the datasets and add hf_token)
3. start the training-> train.py (adds a new language 'uzbek' by default and trains LoRa adapters and a new language embedding, ~2.5% of total model weights)
4. test your ready model by running-> test.py (change the text inside)
