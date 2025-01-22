from scripts.swin_transformer_v2 import SwinTransformerV2
import torch
import torch.nn as nn

def build_model(config):
    # Initialise Backbone
    model = SwinTransformerV2(  img_size=config.DATA.IMG_SIZE,
                                patch_size=config.BACKBONE.SWINV2.PATCH_SIZE,
                                in_chans=config.BACKBONE.SWINV2.IN_CHANS,
                                num_classes=config.BACKBONE.NUM_CLASSES,
                                embed_dim=config.BACKBONE.SWINV2.EMBED_DIM,
                                depths=config.BACKBONE.SWINV2.DEPTHS,
                                num_heads=config.BACKBONE.SWINV2.NUM_HEADS,
                                window_size=config.BACKBONE.SWINV2.WINDOW_SIZE,
                                mlp_ratio=config.BACKBONE.SWINV2.MLP_RATIO,
                                qkv_bias=config.BACKBONE.SWINV2.QKV_BIAS,
                                drop_rate=config.BACKBONE.DROP_RATE,
                                drop_path_rate=config.BACKBONE.DROP_PATH_RATE,
                                ape=config.BACKBONE.SWINV2.APE,
                                patch_norm=config.BACKBONE.SWINV2.PATCH_NORM,
                                use_checkpoint=config.BACKBONE.USE_CHECKPOINT,
                                pretrained_window_sizes=config.BACKBONE.SWINV2.PRETRAINED_WINDOW_SIZES)
    # Load imagenet weights onto the backbone
    if config.BACKBONE.PRETRAINED is not None:
        load_pretrained(config, model)
    
    # Change classifier for the CVS task (3 classes)
    model.head = nn.Linear(in_features=1024, out_features=3, bias=True)

    # Load backbone weights finetuned on Endoscapes2023 (if required)
    if config.MODEL.ENDOSCAPES_PRETRAINED is not None:
            model.load_state_dict(torch.load(config.MODEL.ENDOSCAPES_PRETRAINED))
    
    if config.MODEL.LSTM:
        # Remove MLP classifier
        model.head = nn.Identity()

        if config.MODEL.E2E != True:
            # Freeze the backbone weights
            for param in model.parameters():
                param.requires_grad = False

        swincvs = SwinLSTMModel(  model,
                                        lstm_hidden_size = config.MODEL.LSTM_PARAMS.HIDDEN_SIZE,
                                        num_lstm_layers=config.MODEL.LSTM_PARAMS.NUM_LAYERS,
                                        multiclassifier = config.MODEL.MULTICLASSIFIER,
                                        inference = config.MODEL.INFERENCE)
        return swincvs

    return model

def load_pretrained(config, model):
    print(f"Loading backbone weight {config.BACKBONE.PRETRAINED}")
    checkpoint = torch.load(config.BACKBONE.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            print("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            print(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)

    print(f"Backbone weights loaded successfully!\n")

    del checkpoint
    torch.cuda.empty_cache()

class SwinLSTMModel(nn.Module):
    def __init__(self, swinv2_model, lstm_hidden_size=256, num_lstm_layers=1, num_classes=3, multiclassifier = False, inference=False):
        super(SwinLSTMModel, self).__init__()
        self.swinv2_model = swinv2_model
        self.lstm_hidden_size = lstm_hidden_size
        self.num_classes = num_classes
        self.multiclassifier = multiclassifier # Toggle for additional classifier after the backbone
        self.inference = inference  # Toggle for inference mode
        
        # LSTM for temporal sequence processing
        self.lstm = nn.LSTM(input_size=self.swinv2_model.num_features,
                            hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers,
                            batch_first=True)
        
        # Fully connected layer for classification after LSTM
        self.fc_lstm = nn.Linear(lstm_hidden_size, num_classes)
        if multiclassifier:
            # New fully connected layer for mid-stream classification (SwinV2 feature classification)
            self.fc_swin = nn.Linear(self.swinv2_model.num_features, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len=5, 3, 384, 384)
        batch_size, seq_len, _, _, _ = x.size()

        # Reshape input for SwinV2 (batch_size * seq_len, 3, 384, 384)
        x = x.view(-1, 3, 384, 384)
        
        # Extract features from SwinV2
        features = self.swinv2_model.forward_features(x)  # Shape: (batch_size * seq_len, num_features)
        
        # Optional mid-stream classification
        if self.multiclassifier and not self.inference:
            swin_classification = self.fc_swin(features)  # Shape: (batch_size * seq_len, num_classes)
            swin_classification = swin_classification.view(batch_size, seq_len, -1)  # Reshape for sequence output
        
        # Reshape back to (batch_size, seq_len, num_features)
        features = features.view(batch_size, seq_len, -1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # Shape: (batch_size, seq_len, hidden_size)
        
        # Use the last time step's output from LSTM for classification
        lstm_classification = self.fc_lstm(lstm_out[:, -1, :])  # Shape: (batch_size, num_classes)

        if self.multiclassifier and not self.inference:
            return swin_classification[:, -1, :], lstm_classification
        else:
            return lstm_classification
        
