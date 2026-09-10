import pixeltable as pxt
import pixeltable.functions as pxtf
from typing import Dict, Any

DEFAULT_MAX_TOKENS = 1024


def get_vision_function(provider: str):
    if provider == "openai":
        try:
            import openai  # noqa: F401
            from pixeltable.functions.openai import chat_completions
        except ImportError:
            raise ImportError("Please install the openai package. pip install openai.")
        return chat_completions
    elif provider == "anthropic":
        try:
            import anthropic  # noqa: F401
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
    # Both providers take the image inline as base64; b64_encode makes that explicit
    # rather than relying on how a raw ColumnRef happens to serialize.
    encoded = pxtf.image.b64_encode(image_col, "png")
    model_kwargs = dict(llm_kwargs or {})

    if provider == "openai":
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + encoded},
                        },
                    ],
                }
            ],
            "model": model,
            "model_kwargs": model_kwargs or None,
        }

    elif provider == "anthropic":
        max_tokens = model_kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": encoded,
                            },
                        },
                    ],
                }
            ],
            "model": model,
            "max_tokens": max_tokens,
            "model_kwargs": model_kwargs or None,
        }

    else:
        raise ValueError(f"Unsupported vision provider: {provider}")


def create_vision_computed_column(
    provider: str, col_name: str, vision_func, vision_args: dict, target_obj: pxt.Table
) -> str:
    description_col_name = f"{col_name}_description"
    response_col = f"{col_name}_response"

    target_obj.add_computed_column(
        **{response_col: vision_func(**vision_args)}, if_exists="ignore"
    )
    response = getattr(target_obj, response_col)

    if provider == "openai":
        description = response.choices[0].message.content
    elif provider == "anthropic":
        description = response.content[0].text
    else:
        raise ValueError(f"Unsupported vision provider: {provider}")

    # Both providers return Json, so the extracted field is Json too. An embedding
    # index needs a String column, hence the cast.
    target_obj.add_computed_column(
        **{description_col_name: description.astype(pxt.String)}, if_exists="ignore"
    )

    return description_col_name
