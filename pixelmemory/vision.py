import logging
import pixeltable as pxt
from .memory import Memory

logger = logging.getLogger(__name__)

def get_vision_function(memory_instance: 'Memory'):
    memory_instance.logger.info(f"Getting vision function for provider: {memory_instance.vision_provider}")
    if memory_instance.vision_provider == "openai":
        try:
            import openai
            from pixeltable.functions.openai import vision
        except ImportError:
            logger.error("Please install the openai package. pip install openai.")
            raise
        return vision
    elif memory_instance.vision_provider == "anthropic":
        try:
            import anthropic
            from pixeltable.functions.anthropic import messages
        except ImportError:
            logger.error("Please install the anthropic package. pip install anthropic.")
            raise
        return messages
    else:
        logger.error(f"Unsupported vision provider: {memory_instance.vision_provider}")
        raise ValueError(f"Unsupported vision provider: {memory_instance.vision_provider}")

def prepare_vision_args(memory_instance: 'Memory', col_name: str, target_obj: pxt.Table) -> dict:
    image_col = getattr(target_obj, col_name)
    
    memory_instance.logger.info(f"Preparing vision arguments for provider: {memory_instance.vision_provider}")
    if memory_instance.vision_provider == "openai":
        args = {
            "prompt": memory_instance.vision_prompt,
            "image": image_col,
            "model": memory_instance.vision_model
        }
        if memory_instance.vision_kwargs:
            args["model_kwargs"] = memory_instance.vision_kwargs
        return args
                    
    elif memory_instance.vision_provider == "anthropic":
        args = {
            "messages": [{
                "role": "user", 
                "content": [
                    {"type": "text", "text": memory_instance.vision_prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_col}}
                ]
            }],
            "model": memory_instance.vision_model
        }
        if memory_instance.vision_kwargs:
            args.update(memory_instance.vision_kwargs)
        return args
        
    else:
        logger.error(f"Unsupported vision provider: {memory_instance.vision_provider}")
        raise ValueError(f"Unsupported vision provider: {memory_instance.vision_provider}")

def create_vision_computed_column(
    memory_instance: 'Memory', col_name: str, vision_func, vision_args: dict, target_obj: pxt.Table
) -> str:
    description_col_name = f"{col_name}_description"
    memory_instance.logger.info(f"Creating computed column '{description_col_name}' for vision description.")
    
    if memory_instance.vision_provider == "openai":
        target_obj.add_computed_column(
            **{description_col_name: vision_func(**vision_args)},
            if_exists="ignore"
        )
    elif memory_instance.vision_provider == "anthropic":
        response_col = f"{col_name}_response"
        target_obj.add_computed_column(
            **{response_col: vision_func(**vision_args)},
            if_exists="ignore"
        )
        target_obj.add_computed_column(
            **{description_col_name: getattr(target_obj, response_col).content[0].text},
            col_type=pxt.String,
            if_exists="ignore"
        )
    
    memory_instance.logger.info(f"Successfully created computed column '{description_col_name}'.")
    return description_col_name
