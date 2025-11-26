import json
import torch


def get_best_device():
    """Auto-detect the best available device: MPS (Mac GPU), CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class DefaultConfig:
    def __init__(self, project_root, config_path=None):
        self.K = 1
        self.project_root = project_root
        self.input_dim = 16
        self.time_kernel = 3

        self.periods = 300
        self.hidden_dim = 256
        self.nodes_in = [234, 57, 20]
        self.nodes_out = [57, 20, 10]
        self.num_blocks = 2
        self.strides = [1 for i in range(self.num_blocks)]
        self.learning_rate = 0.001
        self.batch_size = 4  # Reduced for MPS memory constraints
        self.num_epochs = 100
        self.num_workers = 4  # Number of data loading worker processes
        self.pin_memory = True  # Pin memory for faster GPU transfers (disable for MPS)
        self.model_save_path = project_root / 'models/'
        self.data_path = project_root / 'data/'
        self.log_path = project_root / 'logs/'
        self.device = get_best_device()
        self.seed = 42
        self.num_pitches = 128
        self.time_step = 0.125

        # Autoregressive bottleneck settings (defaults)
        self.use_autoregressive_bottleneck = False
        self.bottleneck_block_type = 'lightweight'

        # Graph structure options
        self.connect_simultaneous_pitches = False  # If True, create complete graph between pitches at same timestep
        self.use_global_graph = True  # If True, use single global graph; if False, use per-timestep graphs (slower)

        # Voice dropout for harmonization learning
        self.voice_dropout_rate = 0.2  # Probability of masking each voice node during training

        # Optionally override with config file
        if config_path:
            self.load_from_file(config_path)
        self.load_composer_voice_map()
        self.display()

    def load_from_file(self, path):
        import yaml
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
        for k, v in params.items():
            setattr(self, k, v)

    def display(self):
        print("Configuration:")
        print(f"K: {self.K}")
        print(f"num_nodes: {self.num_nodes}")
        print(f"input_dim: {self.input_dim}")
        print(f"time_kernel: {self.time_kernel}")
        print(f"strides: {self.strides}")
        print(f"period: {self.periods}")
        print(f"hidden_dim: {self.hidden_dim}")
        print(f"num_blocks: {self.num_blocks}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Number of Epochs: {self.num_epochs}")
        print(f"Num Workers: {self.num_workers}")
        print(f"Pin Memory: {self.pin_memory}")
        print(f"Model Save Path: {self.model_save_path}")
        print(f"Data Path: {self.data_path}")
        print(f"Log Path: {self.log_path}")
        print(f"Device: {self.device}")
        print(f"Seed: {self.seed}")
        print(f"num_composers: {self.num_composers}")
        print(f"max_voices: {self.max_voices}")
        print(f"use_autoregressive_bottleneck: {self.use_autoregressive_bottleneck}")
        print(f"bottleneck_block_type: {self.bottleneck_block_type}")
        print(f"connect_simultaneous_pitches: {self.connect_simultaneous_pitches}")
        print(f"use_global_graph: {self.use_global_graph}")
        print(f"voice_dropout_rate: {self.voice_dropout_rate}") 

    def get_rhythm_label(self, rhythm_value):
        rhythm_value = min(float(rhythm_value), 5.0)
        return min(int((rhythm_value / 0.125)), self.max_rhythm)
    
    def load_composer_voice_map(self):
        # Load composer mapping
        print(self.project_root)
        print(self.data_path)
        print(f'loading composer mapping default config')
        with open(f"{self.project_root / self.data_path / 'raw_test/composer_map.json'}", 'r') as f:
            self.composer_map = json.load(f)
            self.num_composers = len(self.composer_map)

        print(f'loading processing statistics default config')

        # Load processing statistics to get max voices
        with open(f"{self.project_root / self.data_path / 'raw_test/processing_stats.json'}", 'r') as f:
            stats = json.load(f)
            self.max_voices = stats["max_voices_used"]
        





        self.pitch_feature_dim = self.num_pitches  # pitch, velocity, voice_id
        self.max_pitch = self.num_pitches - 1
        self.voice_feature_dim = self.max_voices  # voice_id
        self.num_voices= self.max_voices + 1
        self.composer_feature_dim = self.num_composers   # composer_id
        self.max_rhythm = 39
        self.num_rhythms = self.max_rhythm + 1
        self.total_node_feature_dim = self.num_pitches + self.num_voices + self.num_rhythms  # pitches + voices + one rhythm symbol per voice + single composer
        self.num_nodes = self.total_node_feature_dim
        self.num_labels = self.total_node_feature_dim + self.composer_feature_dim


    def get_composer_name(self, composer_id):
        """Get composer name from ID"""
        for name, idx in self.composer_map.items():
            if idx == composer_id:
                return name
        return "Unknown"