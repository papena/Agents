import requests

def load_mcp_tools(base_url: str):
    response = requests.get(f"{base_url}/tools")
    tools_data = response.json()["tools"]
    
    return tools_data


from langchain.tools import StructuredTool
from pydantic import create_model

def build_tools(base_url: str):
    tools_data = load_mcp_tools(base_url)
    tools = []

    for t in tools_data:
        name = t["name"]
        description = t["description"]
        schema = t["input_schema"]["properties"]

        # Dynamically create Pydantic model
        fields = {
            k: (str, None) for k in schema.keys()
        }

        InputModel = create_model(f"{name}_input", **fields)

        def make_func(tool_name):
            def _func(**kwargs):
                return requests.post(
                    f"{base_url}/tools/{tool_name}",
                    json=kwargs
                ).json()
            return _func

        tool = StructuredTool.from_function(
            func=make_func(name),
            name=name,
            description=description,
            args_schema=InputModel
        )

        tools.append(tool)

    return tools