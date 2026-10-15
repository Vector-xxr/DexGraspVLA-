# Standard library imports
import base64
import socket
import sys
import time
import select
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from urllib.parse import urlparse

# Third-party library imports
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from scipy.interpolate import CubicSpline
from rich.console import Console
from rich.text import Text


LOG_COLOR_SCHEME = {
    "timestamp": "#4EC9B0",          # aqua green
    "info": "#CE9178",               # warm orange
    "outcome": "#C586C0",            # muted purple
    "request_task": "#DCDCAA",       # parchment yellow
    "planner_prompt": "#569CD6",     # cool blue
    "planner_response": "#D7BA7D",   # sand brown
    "execution_time": "#808080",     # neutral gray
}
console = Console()


@contextmanager
def timer(name, log_file = None):
    start = datetime.now()
    yield
    end = datetime.now()
    log(f"Execution time of <{name}>: {(end - start).total_seconds():.2f} second(s).", message_type="execution_time", log_file=log_file)


def log(message: str, message_type: str = None, log_file=None):
    # 获取时间戳和样式
    timestamp_raw = time.strftime("[%Y-%m-%d %H:%M:%S]")
    timestamp_color = LOG_COLOR_SCHEME["timestamp"]
    message_color = LOG_COLOR_SCHEME.get(message_type, "white")

    # 写入纯文本文件
    if log_file is not None:
        log_file.write(f"{timestamp_raw} {message}\n")
        log_file.flush()

    # 构建 rich Text 对象
    text = Text()
    text.append(f"{timestamp_raw} ", style=timestamp_color)

    # 多行处理 + 空行保留
    for idx, line in enumerate(message.splitlines()):
        if idx > 0:
            text.append("\n")  # 保持多行结构
        text.append(line, style=message_color)  # 空行也保留颜色

    console.print(text)


def load_config(main_config_path, task_config_path):
    """
    Load main configuration file and its referenced configuration files

    Args:
        config_path: Configuration file root directory
        config_name: Main configuration file name (without .yaml)
    """

    # Create default configuration
    default_cfg = OmegaConf.create({
        "hydra": {
            "job": {
                "num": 0,  # Provide default value
                "override_dirname": "${name}"
            }
        }
    })

    # Load main configuration file
    cfg = OmegaConf.load(main_config_path)

    # Merge default configuration
    cfg = OmegaConf.merge(default_cfg, cfg)
    task_cfg = OmegaConf.load(task_config_path)
    cfg["task"] = task_cfg

    # Parse all variable references
    OmegaConf.resolve(cfg)

    return cfg


def encode_image_to_base64(image):
    if isinstance(image, Image.Image):
        image_pil = image
    elif isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        raise ValueError(f"Invalid image type: {type(image)}")
    
    buffer = BytesIO()
    image_pil.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    return base64_str


def get_image_url(image):
    base64_str = encode_image_to_base64(image)
    image_url = f"data:image/png;base64,{base64_str}"
    return image_url


def decode_base64_to_image(base64_string):
    padding = 4 - (len(base64_string) % 4)
    if padding != 4:
        base64_string += '=' * padding
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    img_data = base64.b64decode(base64_string)
    bytes_io = BytesIO(img_data)
    image = Image.open(bytes_io)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def preprocess_img(image):
    height, width = image.shape[:2]
    processed_image = np.ones_like(image) * 255
    
    # Calculate crop area based on image ratio
    h_start = int(height * 0.0)
    h_end = int(height * 0.6)
    w_start = int(width * 0.4)
    w_end = int(width * 0.9)
    
    processed_image[h_start:h_end, w_start:w_end] = image[h_start:h_end, w_start:w_end]
    return processed_image


def cubic_spline_interpolation_7d(points, step=0.01):
    # Ensure input points have correct dimensions and quantity
    assert points.shape[1] == 7, "Input points must be 7-dimensional"
    num_points = points.shape[0]

    # Build time variable
    t = np.linspace(0, 1, num_points)
    
    # Create cubic spline interpolation for each dimension
    cs_list = [CubicSpline(t, points[:, i]) for i in range(7)]
    
    # Generate interpolation points
    t_interp = np.arange(0, 1, step)
    interpolated_points = np.array([cs(t_interp) for cs in cs_list]).T

    return interpolated_points


def clear_input_buffer():
    # Clear input buffer using different methods for better compatibility
    try:
        # For Windows
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        # For Unix/Linux/MacOS
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        # Fallback method
        while select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.read(1)


def get_start_command():
    clear_input_buffer()
    while True:
        user_input = input("Press <Enter> to start, <q> to quit.")
        if user_input == 'q':
            return False
        elif user_input == '':
            return True
        else:
            print("Invalid input. Please press <Enter> or <q>.")


def show_mask(mask, ax, color):
    color_rgba = np.array(color)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color_rgba.reshape(1, 1, -1)
    ax.imshow(mask_image)


def update_array(existing_array, new_array):
    # Create new array to store updated data
    updated_array = np.empty_like(existing_array)

    # Move the previous array's last item to the second position
    for i in range(0, existing_array.shape[0]):
        if i < existing_array.shape[0]-1:
            updated_array[i, ...] = existing_array[i+1, ...]
        else:
            # Add new array to the last position of the first dimension
            updated_array[i, ...] = new_array

    return updated_array


def check_url(url: str, timeout: int = 5) -> bool:
    """
    Check if URL is accessible
    
    Args:
        url: URL to check
        timeout: Timeout in seconds
    
    Returns:
        bool: Whether URL is accessible
    """
    try:
        parsed_url = urlparse(url)
        host = parsed_url.hostname
        port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
        
        # Try to establish socket connection
        socket.create_connection((host, port), timeout=timeout)
        return True
        
    except Exception as e:
        print(f"Unable to connect to URL: {url}")
        return False