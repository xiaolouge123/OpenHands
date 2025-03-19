from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

# from browsergym/core/action/highlevel.py
# _browser_action_space = HighLevelActionSet(
#     subsets=['bid', 'nav'],
#     strict=False,  # less strict on the parsing of the actions
#     multiaction=True,  # enable to agent to take multiple actions at once
# )


_BROWSER_DESCRIPTION = """Interact with the browser using Python code. Use it ONLY when you need to interact with a webpage and listen the page network activity at the same time.

See the description of "code" parameter for more details.

Multiple actions can be provided at once, but will be executed sequentially without any feedback from the page.
More than 2-3 actions usually leads to failure or unexpected behavior. Example:
goto_and_listen('http://www.example.com')
click_and_listen('a51')
click_and_listen('48', button='middle', modifiers=['Shift'])
"""

_BROWSER_TOOL_DESCRIPTION = """
The following 3 functions are available. Nothing else is supported.

goto_and_listen(url: str)
    Description: Navigate to a url and listen to the page content.
    Examples:
        goto_and_listen('http://www.example.com')


scroll_and_listen(delta_x: float, delta_y: float)
    Description: Scroll horizontally and vertically. Amounts in pixels, positive for right or down scrolling, negative for left or up scrolling. Dispatches a wheel event.
    Examples:
        scroll_and_listen(0, 200)

        scroll_and_listen(-50.2, -100.5)

click_and_listen(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
    Description: Click an element.
    Examples:
        click_and_listen('a51')

        click_and_listen('b22', button='right')

        click_and_listen('48', button='middle', modifiers=['Shift'])

"""


# for _, action in _browser_action_space.action_set.items():
#     assert (
#         action.signature in _BROWSER_TOOL_DESCRIPTION
#     ), f'Browser description mismatch. Please double check if the BrowserGym updated their action space.\n\nAction: {action.signature}'
#     assert (
#         action.description in _BROWSER_TOOL_DESCRIPTION
#     ), f'Browser description mismatch. Please double check if the BrowserGym updated their action space.\n\nAction: {action.description}'

BrowserListenTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='browser_listen',
        description=_BROWSER_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'code': {
                    'type': 'string',
                    'description': (
                        'The Python code that interacts with the browser.\n'
                        + _BROWSER_TOOL_DESCRIPTION
                    ),
                }
            },
            'required': ['code'],
        },
    ),
)
