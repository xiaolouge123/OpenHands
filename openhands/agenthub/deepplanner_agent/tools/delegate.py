from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_DELEGATE_DESCRIPTION = """Delegate the task to a specific agent.
* The assistant can delegate the task to a specific agent by specifying the agent name and the inputs.
* Split a big task into several smaller tasks, even though the smaller tasks share the same pattern but have different inputs.
* The agent name should be one of the following: `DeepCoderAgent`, `BrowserMonitorAgent`.
* `BrowserMonitorAgent` is suitable for task about browsing the web and interacting with the browser to complete multi-step tasks.
* `DeepCoderAgent` is suitable for writing code, editing files, and running commands to complete coding tasks, but with no internet access.
"""

DelegateTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='delegate',
        description=_DELEGATE_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'agent': {
                    'type': 'string',
                    'description': 'The name of the agent to delegate the task to.',
                },
                'inputs': {
                    'type': 'object',
                    'description': 'The task inputs to the agent. Inputs must contain `task` field to specify the task needed to be done and `working_dir` field to specify the working directory.',
                },
            },
            'required': ['agent', 'inputs'],
        },
    ),
)
