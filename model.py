import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

# ============================================
# 1. SWIN TRANSFORMER BLOCK
# ============================================

def window_partition(x, window_size):
    """
    Partitions the input into windows
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, [-1, H//window_size, window_size, W//window_size, window_size, C])
    windows = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(windows, [-1, window_size, window_size, C])
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Merges windows back to original feature map
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = tf.reshape(windows, [B, H//window_size, W//window_size, window_size, window_size, -1])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, -1])
    return x

class WindowAttention(Layer):
    """
    Window based multi-head self attention (W-MSA) module
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0., **kwargs):
        super(WindowAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
    def build(self, input_shape):
        self.qkv = Dense(self.dim * 3, use_bias=True)
        self.attn_drop = Dropout(0.0)
        self.proj = Dense(self.dim)
        self.proj_drop = Dropout(0.0)
        super().build(input_shape)
    
    def call(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [-1, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = tf.reshape(attn, [-1, nW, self.num_heads, N, N]) + mask
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])
        
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def swin_transformer_block(x, dim, num_heads, window_size=7, shift_size=0):
    """
    Complete Swin Transformer Block as per paper
    """
    B, H, W, C = x.shape
    shortcut = x
    
    # Layer normalization
    x = LayerNormalization(epsilon=1e-5)(x)
    
    # Reshape if using shifted windows
    if shift_size > 0:
        shifted_x = tf.roll(x, shift=[-shift_size, -shift_size], axis=[1, 2])
        windows = window_partition(shifted_x, window_size)
    else:
        windows = window_partition(x, window_size)
    
    # Multi-head attention
    windows = tf.reshape(windows, [-1, window_size * window_size, C])
    attn_windows = WindowAttention(dim, window_size, num_heads)(windows)
    attn_windows = tf.reshape(attn_windows, [-1, window_size, window_size, C])
    
    # Reverse window partition
    if shift_size > 0:
        x = window_reverse(attn_windows, window_size, H, W)
        x = tf.roll(x, shift=[shift_size, shift_size], axis=[1, 2])
    else:
        x = window_reverse(attn_windows, window_size, H, W)
    
    # First residual connection
    x = Add()([shortcut, x])
    
    # MLP
    x_norm = LayerNormalization(epsilon=1e-5)(x)
    mlp = Dense(dim * 4, activation='gelu')(x_norm)
    mlp = Dropout(0.0)(mlp)
    mlp = Dense(dim)(mlp)
    mlp = Dropout(0.0)(mlp)
    
    # Second residual connection
    x = Add()([x, mlp])
    
    return x

# ============================================
# CFE BLOCK (Cross-Fusion Enhancement)
# ============================================

def CFE_block(high_res, low_res, growth_rate=32, bias=True):
    """
    CFE Block as per paper equations (2) and (3)
    """
    # High resolution stream
    high_maps = [high_res]
    low_maps = [low_res]
    
    for d in range(1, 6):
        # For high resolution stream
        if d == 1:
            high_input = high_maps[d-1]
        else:
            # Concatenate all previous maps 
            concat_list = []
            for i in range(d):
                if i == d-1:
                    # Upsample low resolution from previous step
                    low_up = Conv2DTranspose(growth_rate, 3, strides=2, padding='same')(low_maps[d-2])
                    concat_list.append(low_up)
                else:
                    concat_list.append(high_maps[i])
            high_input = Concatenate()(concat_list)
        
        high_map = Conv2D(growth_rate, 3, padding='same', use_bias=bias)(high_input)
        high_map = LeakyReLU(alpha=0.25)(high_map)
        high_maps.append(high_map)
        
        # For low resolution stream
        if d == 1:
            low_input = low_maps[d-1]
        else:
            # Concatenate all previous maps 
            concat_list = []
            for i in range(d):
                if i == d-1:
                    # Downsample high resolution 
                    high_down = Conv2D(growth_rate, 3, strides=2, padding='same')(high_maps[d-2])
                    concat_list.append(high_down)
                else:
                    concat_list.append(low_maps[i])
            low_input = Concatenate()(concat_list)
        
        low_map = Conv2D(growth_rate, 3, padding='same', use_bias=bias)(low_input)
        low_map = LeakyReLU(alpha=0.25)(low_map)
        low_maps.append(low_map)
    
    # Residual scaling 
    w = 0.4  # Scaling factor
    high_output = Add()([high_res, Lambda(lambda x: x * w)(high_maps[-1])])
    low_output = Add()([low_res, Lambda(lambda x: x * w)(low_maps[-1])])
    
    return high_output, low_output

# ============================================
# 3. MULTI-ATTENTION GATE (MAG)
# ============================================

def MultiAttentionGate(E, g, channels):
    """
    Attention Gate as per paper equations (6) and (7)
    """
    # Linear transformations
    theta_E = Conv2D(channels, 1, padding='same')(E)
    phi_g = Conv2D(channels, 1, padding='same')(g)
    
    # Additive attention as per paper
    q_att = Add()([theta_E, phi_g])
    q_att = Activation('relu')(q_att)
    
    # Attention coefficients
    q_att = Conv2D(1, 1, padding='same')(q_att)
    attention_coeff = Activation('sigmoid')(q_att)  # Eq. (9) sigmoid
    
    # Apply attention
    E_hat = Multiply()([E, attention_coeff])
    
    return E_hat

# ============================================
# 4. MAIN TCFMA-NET ARCHITECTURE
# ============================================

def TCFMA_Net(input_size=(256, 256, 3), input_size_2=(256, 256, 1)):
    """
    Main network architecture aligned with paper methodology
    """
    inputs_img = Input(input_size)
    canny = Input(input_size_2, name='canny_edge')
    
    # ============================
    # Transformer Branch (Swin Transformers)
    # ============================
    print("Building Transformer Branch...")
    
    # Patch partition and linear embedding
    x = Conv2D(96, 4, strides=4, padding='same')(inputs_img)  # Patch partition
    x = LayerNormalization(epsilon=1e-5)(x)
    
    # Four Swin Transformer stages 
    # Stage 1
    x = swin_transformer_block(x, dim=96, num_heads=3, window_size=7, shift_size=0)
    x = swin_transformer_block(x, dim=96, num_heads=3, window_size=7, shift_size=3)
    stage1_out = x  # C2
    
    # Stage 2 (downsample)
    x = Conv2D(192, 2, strides=2, padding='same')(x)
    x = swin_transformer_block(x, dim=192, num_heads=6, window_size=7, shift_size=0)
    x = swin_transformer_block(x, dim=192, num_heads=6, window_size=7, shift_size=3)
    stage2_out = x  # C3
    
    # Stage 3 (downsample)
    x = Conv2D(384, 2, strides=2, padding='same')(x)
    x = swin_transformer_block(x, dim=384, num_heads=12, window_size=7, shift_size=0)
    x = swin_transformer_block(x, dim=384, num_heads=12, window_size=7, shift_size=3)
    stage3_out = x  # C4
    
    # Stage 4 (downsample)
    x = Conv2D(768, 2, strides=2, padding='same')(x)
    x = swin_transformer_block(x, dim=768, num_heads=24, window_size=7, shift_size=0)
    x = swin_transformer_block(x, dim=768, num_heads=24, window_size=7, shift_size=3)
    stage4_out = x  # C5
    
    # ============================
    # CFE Network with CFE Blocks
    # ============================
    print("Building CFE Network...")
    
    # Initial CNN encoder to match transformer feature sizes
    # Encoder path
    e1 = Conv2D(64, 3, padding='same')(inputs_img)
    e1 = BatchNormalization()(e1)
    e1 = Activation('relu')(e1)
    e1 = Conv2D(64, 3, padding='same')(e1)
    e1 = BatchNormalization()(e1)
    e1 = Activation('relu')(e1)
    p1 = MaxPooling2D(2)(e1)
    
    e2 = Conv2D(128, 3, padding='same')(p1)
    e2 = BatchNormalization()(e2)
    e2 = Activation('relu')(e2)
    e2 = Conv2D(128, 3, padding='same')(e2)
    e2 = BatchNormalization()(e2)
    e2 = Activation('relu')(e2)
    p2 = MaxPooling2D(2)(e2)
    
    e3 = Conv2D(256, 3, padding='same')(p2)
    e3 = BatchNormalization()(e3)
    e3 = Activation('relu')(e3)
    e3 = Conv2D(256, 3, padding='same')(e3)
    e3 = BatchNormalization()(e3)
    e3 = Activation('relu')(e3)
    p3 = MaxPooling2D(2)(e3)
    
    e4 = Conv2D(512, 3, padding='same')(p3)
    e4 = BatchNormalization()(e4)
    e4 = Activation('relu')(e4)
    e4 = Conv2D(512, 3, padding='same')(e4)
    e4 = BatchNormalization()(e4)
    e4 = Activation('relu')(e4)
    
    # Fuse transformer features with CNN features
    # Resize transformer features to match CNN feature sizes
    stage1_out = Conv2D(64, 1, padding='same')(stage1_out)
    stage2_out = Conv2D(128, 1, padding='same')(stage2_out)
    stage3_out = Conv2D(256, 1, padding='same')(stage3_out)
    stage4_out = Conv2D(512, 1, padding='same')(stage4_out)
    
    # Apply CFE blocks at multiple scales
    cfe1_high, cfe1_low = CFE_block(e1, e2)
    cfe2_high, cfe2_low = CFE_block(e2, e3)
    cfe3_high, cfe3_low = CFE_block(e3, e4)
    
    # ============================
    #Multi-Attention Modules
    # ============================
    print("Building Multi-Attention Modules...")
    
    # Apply attention gates to skip connections
    
    att1 = MultiAttentionGate(cfe1_high, stage4_out, 64)
    att2 = MultiAttentionGate(cfe2_high, stage4_out, 128)
    att3 = MultiAttentionGate(cfe3_high, stage4_out, 256)
    
    # ============================
    # 3.4 Decoder of TCFMA Network
    # ============================
    print("Building Decoder...")
    
    # Decoder with skip connections and attention
    # Level 4 to 3
    d4 = Conv2DTranspose(256, 2, strides=2, padding='same')(stage4_out)
    d4 = Concatenate()([d4, att3])
    d4 = Conv2D(256, 3, padding='same')(d4)
    d4 = BatchNormalization()(d4)
    d4 = Activation('relu')(d4)
    
    # Level 3 to 2
    d3 = Conv2DTranspose(128, 2, strides=2, padding='same')(d4)
    d3 = Concatenate()([d3, att2])
    d3 = Conv2D(128, 3, padding='same')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Activation('relu')(d3)
    
    # Level 2 to 1
    d2 = Conv2DTranspose(64, 2, strides=2, padding='same')(d3)
    d2 = Concatenate()([d2, att1])
    d2 = Conv2D(64, 3, padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = Activation('relu')(d2)
    
    # Final convolution to match input size
    d1 = Conv2DTranspose(32, 2, strides=2, padding='same')(d2)
    d1 = Conv2D(32, 3, padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    
    # ============================
    # Edge Stream (Shape Stream)
    # ============================
    print("Building Edge Stream...")
    
    # Edge detection stream as in your original code
    edge_stream = Conv2D(32, 1, padding='same')(d3)
    edge_stream = UpSampling2D(2)(edge_stream)
    edge_stream = Conv2D(16, 3, padding='same')(edge_stream)
    edge_stream = BatchNormalization()(edge_stream)
    edge_stream = Activation('relu')(edge_stream)
    
    # Fuse with Canny edge input
    edge_stream = Concatenate()([edge_stream, canny])
    edge_stream = Conv2D(8, 3, padding='same')(edge_stream)
    edge_stream = BatchNormalization()(edge_stream)
    edge_stream = Activation('relu')(edge_stream)
    
    edge_out = Conv2D(1, 1, padding='same', activation='sigmoid', name='edge_out')(edge_stream)
    
    # Fuse edge information with main decoder
    d1 = Concatenate()([d1, edge_stream])
    d1 = Conv2D(32, 3, padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    
    # ============================
    # Final Output
    # ============================
    
    # Main segmentation output
    main_output = Conv2D(1, 1, padding='same', activation='sigmoid', name='main_output')(d1)
    
    # Auxiliary outputs at different scales (deep supervision)
    aux4 = Conv2D(1, 1, padding='same', activation='sigmoid')(d4)
    aux4 = UpSampling2D(8, interpolation='bilinear')(aux4)
    
    aux3 = Conv2D(1, 1, padding='same', activation='sigmoid')(d3)
    aux3 = UpSampling2D(4, interpolation='bilinear')(aux3)
    
    # ============================
    # Loss Function
    # ============================
    
    model = Model(inputs=[inputs_img, canny], 
                  outputs=[main_output, edge_out, aux3, aux4],
                  name='TCFMA_Net')
    
    print("TCFMA-Net model built successfully!")
    return model

# ============================================
# COMPILE AND TRAIN
# ============================================

def compile_model(model):
    """
    Compile model with paper-specified loss function
    """
    # Loss weights as per paper (adjust as needed)
    loss_weights = {'main_output': 2.0, 'edge_out': 1.0, 'aux3': 1.0, 'aux4': 1.0}
    
    # Combined loss: Binary Cross-Entropy + Dice Loss
    def combined_loss(y_true, y_pred):
        # Binary Cross-Entropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Dice Loss
        smooth = 1.0
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        dice_loss = 1 - dice
        
        # Combined loss with equal weights as per paper Eq. (13)
        return bce + dice_loss
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer,
                  loss={'main_output': combined_loss,
                        'edge_out': 'binary_crossentropy',
                        'aux3': combined_loss,
                        'aux4': combined_loss},
                  loss_weights=loss_weights,
                  metrics={'main_output': ['accuracy', dice_coefficient]})
    
    return model

# ============================================
# UTILITY FUNCTIONS
# ============================================

def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

