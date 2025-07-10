import pixeltable as pxt
from typing import Dict, Any


def get_vision_function(provider: str):
    if provider == "openai":
        try:
            import openai
            from pixeltable.functions.openai import vision
        except ImportError:
            raise ImportError("Please install the openai package. pip install openai.")
        return vision
    elif provider == "anthropic":
        try:
            import anthropic
            from pixeltable.functions.anthropic import messages
        except ImportError:
            raise ImportError(
                "Please install the anthropic package. pip install anthropic."
            )
        return messages
    else:
        raise ValueError(f"Unsupported vision provider: {provider}")


def prepare_vision_args(
    provider: str,
    model: str,
    prompt: str,
    llm_kwargs: Dict[str, Any],
    col_name: str,
    target_obj: pxt.Table,
) -> dict:
    image_col = getattr(target_obj, col_name)

    if provider == "openai":
        args = {"prompt": prompt, "image": image_col, "model": model}
        if llm_kwargs:
            args["model_kwargs"] = llm_kwargs
        return args

    elif provider == "anthropic":
        args = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_col,
                            },
                        },
                    ],
                }
            ],
            "model": model,
        }
        if llm_kwargs:
            args.update(llm_kwargs)
        return args

    else:
        raise ValueError(f"Unsupported vision provider: {provider}")


def create_vision_computed_column(
    provider: str, col_name: str, vision_func, vision_args: dict, target_obj: pxt.Table
) -> str:
    description_col_name = f"{col_name}_description"

    if provider == "openai":
        target_obj.add_computed_column(
            **{description_col_name: vision_func(**vision_args)}, if_exists="ignore"
        )
    elif provider == "anthropic":
        response_col = f"{col_name}_response"
        target_obj.add_computed_column(
            **{response_col: vision_func(**vision_args)}, if_exists="ignore"
        )
        target_obj.add_computed_column(
            **{description_col_name: getattr(target_obj, response_col).content[0].text},
            if_exists="ignore",
        )

    return description_col_name
