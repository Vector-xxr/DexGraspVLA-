import os
import time
import json
import copy
from typing import Union

import httpx
import json_repair
import matplotlib.pyplot as plt
from openai import OpenAI

from planner.utils import parse_json
from inference_utils.utils import decode_base64_to_image, log


class DexGraspVLAPlanner:
    def __init__(self,
                api_key: str = "EMPTY", 
                base_url: str = "http://localhost:8000/v1",
                model_name: str = None):

        transport = httpx.HTTPTransport(retries=1)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(transport=transport)
        )
        self.model = self.client.models.list().data[0].id if model_name is None else model_name
        self.log_file = None
        self.image_dir = None


    def set_logging(self, log_file, image_dir):
        self.log_file = log_file
        self.image_dir = image_dir


    def request_task(self,
            task_name: str,
            vl_inputs: dict[str, Union[str, dict]] = None,
            max_token: int = 512
    ) -> str:
        if task_name == "grasping_instruction_proposal":
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                                f"You are controlling a robotic arm that needs to complete the following user prompt: {vl_inputs['user_prompt']}\n"
                                f"I will show you two images. The initial image (before any actions) is:"
                            )
                        },
                        {"type": "image_url", "image_url": {"url": vl_inputs["images"]["initial_head_image"]}},
                        {"type": "text", "text": "The current image (after the latest action) is:"},
                        {"type": "image_url", "image_url": {"url": vl_inputs["images"]["current_head_image"]}},
                        {"type": "text", "text": (
                                f"Your task is to select the **best object to grasp next** from the current image.\n"
                                f"To identify objects, **use common sense and everyday knowledge** to infer what each item is.\n"
                                f"For example, recognize cups, bottles, fruits, snacks, boxes, tools, etc.\n"

                                f"When choosing the best object to grasp, follow these principles:\n"
                                f"1. Prefer objects on the right, then center, then left.\n"
                                f"2. Avoid objects that are blocked or surrounded.\n"
                                f"3. Avoid grasping objects that would cause other items to topple.\n"
                                f"4. Select objects that best match the user prompt.\n\n"

                                f"Please output ONLY ONE object that the robot should grasp next.\n"

                                f"Return format (in English, natural language):\n"
                                f"- A short sentence precisely describing the target object, including:\n"
                                f"- color\n"
                                f"- shape\n"
                                f"- relative position (e.g., \"on the right\", \"in front\", \"next to the red box\")\n"

                                f"Example:\n"
                                f"Grasp the blue cube on the right side of the table.\n"
                            )
                        }
                    ]
                }
            ]

        elif task_name == "bounding_box_prediction":
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"You are a robotic vision assistant. Your task is to locate the object described below in the given image:"},
                        {"type": "image_url", "image_url": {"url": vl_inputs["images"]["current_head_image"]}},
                        {"type": "text", "text": (
                                f"and return its bounding box.\n\n"
                                f"Grasping instruction: {vl_inputs['grasping_instruction']}\n\n"

                                f"Instructions:\n"
                                f"1. Carefully read the grasping instruction and match the target object to the best-fitting visible object in the image.\n"
                                f"2. Select EXACTLY ONE object that best matches the description.\n"
                                f"3. For the selected object, return the following in strict JSON format:\n"
                                f"   - \"bbox_2d\": [x1, y1, x2, y2]  (integer pixel coordinates, top-left to bottom-right)\n"
                                f"   - \"label\": a short 2-4 word name (e.g., \"blue cup\")\n"
                                f"   - \"description\": a complete, natural-language description of the object's appearance and position\n\n"
                                
                                f"Requirements:\n"
                                f"- Only return one object.\n"
                                f"- Coordinates must be valid and within image boundaries.\n"
                                f"- Do not guess if the object is not visible"
                            )
                        }
                    ]
                }
            ]

        elif task_name == "grasp_outcome_verification":
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "I will show you two images. The top-down view from the head camera is:"},
                        {"type": "image_url", "image_url": {"url": vl_inputs["images"]["current_head_image"]}},
                        {"type": "text", "text": "The close-up view from the wrist camera is:"},
                        {"type": "image_url", "image_url": {"url": vl_inputs["images"]["current_wrist_image"]}},
                        {"type": "text", "text": (
                                f"Grasping instruction: {vl_inputs['grasping_instruction']}\n\n"

                                f"Task:\n"
                                f"Determine whether the robotic arm has **successfully grasped the target object**.\n"

                                f"You should consider:\n"
                                f"- Whether the target object is still visible on the table.\n"
                                f"- Whether the object is securely held in the robotic hand.\n"

                                f"Output format: A reasoning and a boolean value (True=successfully grasped, False=not grasped).\n"
                                
                                f"Keep it short and simple.\n\n"
                            )
                        }
                    ]
                }
            ]

        elif task_name == "prompt_completion_check":
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                                f"The robot is trying to complete the following user prompt: {vl_inputs['user_prompt']}\n"
                                f"I will show you two images. The initial image (before any actions) is:"
                            )
                        },
                        {"type": "image_url", "image_url": {"url": vl_inputs["images"]["initial_head_image"]}},
                        {"type": "text", "text": "The current image (after the latest action) is:"},
                        {"type": "image_url", "image_url": {"url": vl_inputs["images"]["current_head_image"]}},
                        {"type": "text", "text": (
                                f"Please compare the two images and determine whether the user prompt has been fully completed.\n"

                                f"Instructions:\n"
                                f"- Only consider visible 3D objects.\n"
                                f"- If all target objects have been removed or grasped, return True.\n"
                                f"- If some relevant objects remain, return False.\n"

                                f"Output format: A reasoning and a boolean value (True=completed, False=not completed).\n"
                                
                                f"Example:\n"
                                f"All blue objects have been removed from the table: True. \n"
                            )
                        }
                    ]
                }
            ]

        else:
            raise ValueError(f"The task_name {task_name} is not a valid task name.")

        self.log(f"Planner requesting task: {task_name}.", message_type="request_task")
        messages_for_logging = self.process_message_for_logging(copy.deepcopy(messages))
        self.log(f"Planner prompt:\n{messages_for_logging}", message_type="planner_prompt")

        if vl_inputs and "images" in vl_inputs:
            for key, image_url in vl_inputs["images"].items():
                self.save_image(image_url, task_name, key)
        
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_token,
            messages=messages
        )

        response = chat_completion.choices[0].message.content
        response_lower = response.lower()

        self.log(f"Planner response:\n{response}", message_type="planner_response")

        if task_name == "grasping_instruction_proposal":
            # Return the text description of the target object
            return response.strip()
            
        elif task_name == "bounding_box_prediction":
            bbox_str = parse_json(response)
            bbox_json = json_repair.loads(bbox_str)
            return bbox_json
            
        else:
            if 'true' in response_lower:
                return True
            elif 'false' in response_lower:
                return False
            else:
                raise ValueError(f"The output text {response} does not contain a valid boolean value.") 

    def process_message_for_logging(self, messages):
        
        def replace_base64_images(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and value.startswith(('data:image', 'http')):
                        obj[key] = '<omitted>'
                    else:
                        replace_base64_images(value)
            elif isinstance(obj, list):
                for item in obj:
                    replace_base64_images(item)
            return obj
        
        # Process the messages copy
        processed_messages = replace_base64_images(messages)
        
        # Convert to formatted JSON string
        messages_for_logging = json.dumps(processed_messages, indent=2, ensure_ascii=False)
        return messages_for_logging


    def log(self, message, message_type = None):
        log(message, message_type, self.log_file)

    
    def save_image(self, image_url, task_name, image_type):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image = decode_base64_to_image(image_url)
        image_path = os.path.join(self.image_dir, f"{timestamp}_planner_request_{task_name}_{image_type}.png")
        plt.imsave(image_path, image)
        self.log(f"The {image_type} saved when planner requests {task_name}.", message_type="info")
