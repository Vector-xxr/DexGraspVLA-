import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
from skimage.color import rgb2hsv, hsv2rgb


def visualize_attention_maps(pkl_path: str):
    """
    Process attention maps in a single pkl file and generate visualization results in the same directory
    
    Args:
        pkl_path: Path to the pkl file
    """
    # Read attention maps and obs_dict
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        timestep_attention_maps = data['attention_maps']
        obs_dict = data['obs_dict']
    
    # Create output directory (same name as pkl file without extension)
    output_dir = pkl_path.rsplit('.', 1)[0]
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the attention map from first block of first timestep to determine dimensions
    first_timestep = list(timestep_attention_maps.keys())[0]
    first_block_attn = timestep_attention_maps[first_timestep][0]
    batch_size, num_heads, cond_len = first_block_attn.shape
    
    # Process each sample separately
    for sample_idx in range(batch_size):

        # Get observation images
        rgbm = obs_dict['rgbm'][sample_idx][:, [2, 1, 0, 3]] # [T,4,H,W]
        right_cam = obs_dict['right_cam_img'][sample_idx][:, [2, 1, 0]] # [T,3,H,W]

        # Create sample directory
        sample_dir = os.path.join(output_dir, f'sample_{sample_idx}')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Process each timestep
        for timestep in sorted(timestep_attention_maps.keys()):
            # Create timestep directory
            timestep_dir = os.path.join(sample_dir, f'timestep_{timestep}')
            os.makedirs(timestep_dir, exist_ok=True)
            
            # Get attention maps of all blocks at this timestep
            block_attention_maps = timestep_attention_maps[timestep]
            
            # Process each block
            for block_idx, block_attn in enumerate(block_attention_maps):
                # Create block directory
                block_dir = os.path.join(timestep_dir, f'block_{block_idx}')
                os.makedirs(block_dir, exist_ok=True)
                
                # Process each head
                for head_idx in range(num_heads):
                    # Create head directory
                    head_dir = os.path.join(block_dir, f'head_{head_idx}')
                    os.makedirs(head_dir, exist_ok=True)
                    
                    # Get attention map of this head
                    attn = block_attn[sample_idx, head_idx]
                    
                    # Create a large figure - 3 rows 3 columns
                    fig = plt.figure(figsize=(15, 12))  # Increase figure size to fit new layout
                    
                    # Create a shared colorbar norm
                    token_attn = attn[None, :]
                    vmin = token_attn.min().item()
                    vmax = token_attn.max().item()
                    norm = plt.Normalize(vmin=vmin, vmax=vmax)
                    
                    # First row shows original images
                    # RGB part of rgbm
                    ax1 = plt.subplot(3, 3, 1)
                    rgb = rgbm[0,:3].transpose(1,2,0)
                    plt.imshow(rgb)
                    plt.title('Head Camera RGB')
                    plt.axis('off')
                    
                    # Mask part of rgbm
                    ax2 = plt.subplot(3, 3, 2)
                    mask = rgbm[0,3]
                    plt.imshow(mask, cmap='gray')
                    plt.title('Head Camera Mask')
                    plt.axis('off')
                    
                    # right_cam_img
                    ax3 = plt.subplot(3, 3, 3)
                    right_cam_rgb = right_cam[0].transpose(1,2,0)
                    plt.imshow(right_cam_rgb)
                    plt.title('Wrist Camera')
                    plt.axis('off')
                    
                    # Second row shows attention maps
                    # Attention for first 37x37 image tokens
                    ax4 = plt.subplot(3, 3, 4)
                    head_attn = token_attn[:, :1369].reshape(1, 37, 37)
                    im1 = plt.imshow(head_attn.squeeze(), cmap='viridis', norm=norm, aspect='equal')
                    plt.title('Attention to Head Camera')
                    
                    # Attention for second 37x37 image tokens
                    ax5 = plt.subplot(3, 3, 5)
                    wrist_attn = token_attn[:, 1369:2738].reshape(1, 37, 37)
                    plt.imshow(wrist_attn.squeeze(), cmap='viridis', norm=norm, aspect='equal')
                    plt.title('Attention to Wrist Camera')
                    
                    # Attention for State, Time tokens
                    ax6 = plt.subplot(3, 3, 6)
                    other_tokens = token_attn[:, 2738:].T
                    sns.heatmap(other_tokens, cmap='viridis', square=True, norm=norm,
                                yticklabels=['State', 'Time'], cbar=False)
                    plt.title('Other Tokens')
                    
                    # Third row shows attention overlay
                    # Head Camera attention overlay
                    ax7 = plt.subplot(3, 3, 7)
                    head_attn_scaled = (head_attn - vmin) / (vmax - vmin)
                    head_attn_upsampled = F.interpolate(torch.from_numpy(head_attn_scaled).unsqueeze(0), 
                                                        size=rgb.shape[:2], 
                                                        mode='bilinear').squeeze().numpy()
                    # Convert to HSV and overlay attention on V channel
                    rgb_hsv = rgb2hsv(rgb)
                    rgb_hsv[..., 2] *= head_attn_upsampled
                    rgb_with_attn = hsv2rgb(rgb_hsv)
                    plt.imshow(rgb_with_attn)
                    plt.title('Head Camera Attention Overlay')
                    plt.axis('off')

                    # Wrist Camera attention overlay
                    ax8 = plt.subplot(3, 3, 8)
                    wrist_attn_scaled = (wrist_attn - vmin) / (vmax - vmin)
                    wrist_attn_upsampled = F.interpolate(torch.from_numpy(wrist_attn_scaled).unsqueeze(0), 
                                                            size=right_cam_rgb.shape[:2], 
                                                            mode='bilinear').squeeze().numpy()
                    rgb_hsv = rgb2hsv(right_cam_rgb)
                    rgb_hsv[..., 2] *= wrist_attn_upsampled
                    rgb_with_attn = hsv2rgb(rgb_hsv)
                    plt.imshow(rgb_with_attn)
                    plt.title('Wrist Camera Attention Overlay')
                    plt.axis('off')
                    
                    # Last plot left empty
                    ax9 = plt.subplot(3, 3, 9)
                    plt.axis('off')
                    
                    # Add shared colorbar, adjust position
                    cbar_ax = fig.add_axes([0.95, 0.15, 0.015, 0.7])  # Adjust colorbar position and size
                    fig.colorbar(im1, cax=cbar_ax)
                    
                    # Adjust layout and add overall title
                    plt.suptitle(f'Sample {sample_idx}, Timestep {timestep}, Block {block_idx}, Head {head_idx}, Token 0 Attention',
                                y=0.95)  # Adjust title position
                    plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # Adjust overall layout to leave space for colorbar and title
                    
                    # Save image with higher DPI for better quality
                    plt.savefig(os.path.join(head_dir, f'token_0.png'), 
                                bbox_inches='tight', 
                                dpi=50)  # Increase DPI
                    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize attention maps from pkl files')
    parser.add_argument('--attn_maps_dir', type=str, required=True,
                        help='Directory containing pkl files with attention maps')
    args = parser.parse_args()
    
    # Process all pkl files in the directory
    for pkl_file in sorted(Path(args.attn_maps_dir).glob('*.pkl')):
        print(f"Processing {pkl_file}...")
        visualize_attention_maps(str(pkl_file))

if __name__ == '__main__':
    main()