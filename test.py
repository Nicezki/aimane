from app import aimanev2

if __name__ == '__main__':
    ai = aimanev2.AiMane()
    print(ai.model_config.config)
    print(ai.running_config.config)
    print(ai.prediction_result.config)