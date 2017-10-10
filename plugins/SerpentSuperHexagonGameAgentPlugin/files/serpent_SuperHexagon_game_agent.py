from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey
import offshoot

class SerpentSuperHexagonGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.analytics_client = None

        #Ccontext class setup
        plugin_path = offshoot.config["file_paths"]["plugins"]

        context_classifier_path = "datasets/context_classifier.model"

        from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import \
            CNNInceptionV3ContextClassifier
        context_classifier = CNNInceptionV3ContextClassifier(
            input_shape=(240, 384, 3))  # Replace with the shape (rows, cols, channels) of your captured context frames

        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)

        self.machine_learning_models["context_classifier"] = context_classifier

    def setup_play(self):
        # self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
        pass

    def handle_play(self, game_frame):
        # for i, game_frame in enumerate(self.game_frame_buffer.frames):
        #     self.visual_debugger.store_image_data(
        #         game_frame.frame,
        #         game_frame.frame.shape,
        #         str(i)
        #     )
        # self.input_controller.tap_key(KeyboardKey.KEY_RIGHT)
        context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)
        print("Context:", context)