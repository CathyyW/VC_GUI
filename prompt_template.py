import json

PROMPT_PREFIX = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    " Based on user's goal/request, you may\n"
    '- Answer back if the request/goal is a question (or a chat message),'
    ' like user asks "What is my schedule for today?".\n'
    '- Complete some tasks described in the requests/goals by'
    ' performing actions (step by step) on the phone.\n\n'
    'When given a user request, you will try to complete it step by step.'
    ' At each step, you will be given the current screenshot (including the'
    ' original screenshot and the same screenshot with bounding'
    ' boxes and numeric indexes added to some UI elements) and a history of'
    ' what you have done (in text). Based on these pieces of information and'
    ' the goal, you must choose to perform one of the'
    ' action in the following list (action description followed by the JSON'
    ' format) by outputing the action in the correct JSON format.\n'
    '- If you think the task has been completed, finish the task by using the'
    ' status action with complete as goal_status:'
    ' `{{"action_type": "status", "goal_status": "complete"}}`'
    "- Answer user's question:"
    ' `{{"action_type": "answer", "text": "<answer_text>"}}`\n'
    '- Click/tap on an element on the screen. We have added marks (bounding'
    ' boxes with numeric indexes on their TOP LEFT corner) to most of the UI'
    ' elements in the screenshot, use the numeric index to indicate which'
    ' element you want to click:'
    ' `{{"action_type": "click", "index": <target_index>}}`.\n'
    '- Long press on an element on the screen, similar with the click action'
    ' above, use the numeric label on the bounding box to indicate which'
    ' element you want to long press:'
    ' `{{"action_type": "long_press", "index": <target_index>}}`.\n'
    '- Type text into a text field (this action contains clicking the text'
    ' field, typing in the text and pressing the enter, so no need to click on'
    ' the target field to start), use the numeric label'
    ' on the bounding box to indicate the target text field:'
    ' `{{"action_type": "input_text", "text": <text_input>,'
    ' "index": <target_index>}}`\n'
    '- Press the Enter key: `{{"action_type": "keyboard_enter"}}`\n'
    '- Navigate to the home screen: `{{"action_type": "navigate_home"}}`\n'
    '- Navigate back: `{{"action_type": "navigate_back"}}`\n'
    '- Scroll the screen or a scrollable UI element in one of the four'
    ' directions, use the same numeric index as above if you want to scroll a'
    ' specific UI element, leave it empty when scroll the whole screen:'
    ' `{{"action_type": "scroll", "direction": <up, down, left, right>,'
    ' "index": <optional_target_index>}}`\n'
    ' You may try to scroll the UI element to left/right if you found up/down does not work.\n'
    '- Open an app (nothing will happen if the app is not'
    ' installed): `{{"action_type": "open_app", "app_name": <name>}}`\n'
    '- Wait for the screen to update: `{{"action_type": "wait"}}`\n'
)

PROMPT_PREFIX_V2 = (
    'You are an agent capable of operating an Android phone on behalf of a user. Your task is to assist with user requests or goals by:\n'
    '1. Answering questions or chat-like messages, such as "What is my schedule for today?".\n'
    '2. Performing tasks step-by-step on the phone based on the user\'s instructions.\n\n'
    'For each step, you will be provided:\n'
    '- A history of actions you have taken so far.\n'
    '- The current screenshot\'s HTML description.\n\n'
    'The current date is Sun, Oct 15.\n\n'
)


SELF_EVAL_TEMPLATE_VERIFIER_TRAINING_V3 = (
    '{prompt_prefix}'
    + 'The (overall) user goal/request is: {goal}\n\n'
    'Here is the history of actions taken:\n{history}\n\n'
    'Here is the detailed information about the UI elements in the current screenshot:\n{before_elements}\n'
    '\n'
    'Your task: \n'
    '- Determine if the action is helpful for completing the user\'s task.\n'
    '- Respond with **"Yes"** if the action is helpful, even if it does not directly complete the task.\n'
    '- Respond with **"No"** if the action is not helpful for the task.\n'
    + '\n'
    'Is {action} helpful for completing the task?\n'
    'Answer:'
)


SUMMARY_PROMPT_TEMPLATE_VERIFIER = (
    PROMPT_PREFIX
    + '\nThe (overall) user goal/request is: {goal}\n'
    'Now I want you to summerize the latest step.\n'
    'You will be given the action you chose\n'
    'Also here is the previous screenshot\'s HTML description:\n{before_elements}\n'
    'Here is the current screenshot\'s HTML description:\n{after_elements}\n'
    'This is the action you picked: {action}\n'
    'By comparing the two HTML descriptions and the'
    ' action performed, give a brief summary of this step. This summary'
    ' will be added to action history and used in future action selection,'
    ' so try to include essential information you think that will be most'
    ' useful for future action selections like what you'
    ' intended to do, why, if it worked as expected, if not'
    ' what might be the reason (be critical, the action/reason might be'
    ' wrong), what should/should not be done next and so on. Some more'
    ' rules/tips you should follow:\n'
    '- Keep it short (better less than 50 words) and in a single line\n'
    "- Some actions (like `answer`, `wait`) don't involve screen change,"
    ' you can just assume they work as expected.\n'
    '- Given this summary will be added into action history, it can be used as'
    ' memory to include information that needs to be remembered, or shared'
    ' between different apps.\n\n'
    'Summary of this step: '
)


ACTION_COMPLETION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}\n\n'
    'You may have tried to compolish tasks for several iterations.'
    ' Here is a history of what you have done so far during the past iterations and current iterations:\n{history}\n\n'
    'Here is a list of detailed'
    ' information for some of the UI elements (notice that some elements in'
    ' this list may not be visible in the current screen and so you can not'
    ' interact with it, can try to scroll the screen to reveal it first),'
    ' the numeric indexes are'
    ' consistent with the ones in the labeled screenshot:\n{ui_desc}\n'
    + 'This is the action you select: {action}, which lacks some key information.\n'
    + '\nNow please complete the action in the correct JSON format,'
    ' Action: {action}\n\n'
    'Your Answer:\n'
)


PLANNER_PROMPT_TEMPLATE = (
    'You are the planning agent for an Android automation system.\n\n'
    'Your job is to decompose the user task into short, verifiable subgoals.\n'
    'Do NOT output low-level UI actions such as click index, scroll direction, or coordinates.\n'
    'Each subgoal must be executable by a GUI action agent and must have an observable expected outcome.\n\n'
    'Overall user goal:\n{goal}\n\n'
    'Current screen HTML description:\n{screen_html}\n\n'
    'Execution memory so far:\n{memory}\n\n'
    'Rules:\n'
    '- Produce 2 to 6 subgoals unless the task is already complete or infeasible.\n'
    '- Each subgoal should be concrete and short-horizon.\n'
    '- Each expected_outcome must be checkable from the next screen or recent action history.\n'
    '- Do not assume hidden state that cannot be observed.\n'
    '- If the task is already complete, return an empty subgoals list and task_status="complete".\n'
    '- Output only valid JSON. Do not include markdown fences or extra explanation.\n\n'
    'JSON schema:\n'
    '{{\n'
    '  "task_status": "in_progress" | "complete" | "infeasible",\n'
    '  "overall_strategy": "...",\n'
    '  "subgoals": [\n'
    '    {{\n'
    '      "id": "sg1",\n'
    '      "subgoal": "...",\n'
    '      "expected_outcome": "...",\n'
    '      "max_steps": 3,\n'
    '      "failure_hint": "..."\n'
    '    }}\n'
    '  ]\n'
    '}}\n'
)


REPLAN_PROMPT_TEMPLATE = (
    'You are replanning after a failed Android automation subgoal.\n\n'
    'Overall user goal:\n{goal}\n\n'
    'Previous plan:\n{previous_plan}\n\n'
    'Completed subgoals:\n{completed_subgoals}\n\n'
    'Failed subgoal:\n{failed_subgoal}\n\n'
    'Expected outcome that was not achieved:\n{expected_outcome}\n\n'
    'Execution memory:\n{memory}\n\n'
    'Current screen HTML description:\n{screen_html}\n\n'
    'Analyze why the previous plan failed, then generate a revised plan from the current screen.\n'
    'Treat completed subgoals as already finished; do not include them again in revised_subgoals.\n'
    'Do not repeat the same failed action unless the reason is fixed.\n'
    'If recovery is needed, make the first revised subgoal a recovery subgoal.\n'
    'Do not output low-level UI actions such as click index, scroll direction, or coordinates.\n'
    'Output only valid JSON. Do not include markdown fences or extra explanation.\n\n'
    'JSON schema:\n'
    '{{\n'
    '  "reflection": "...",\n'
    '  "failure_type": "wrong_screen|missing_element|bad_input|premature_finish|loop|unknown",\n'
    '  "revised_subgoals": [\n'
    '    {{\n'
    '      "id": "sg1",\n'
    '      "subgoal": "...",\n'
    '      "expected_outcome": "...",\n'
    '      "max_steps": 3,\n'
    '      "failure_hint": "..."\n'
    '    }}\n'
    '  ]\n'
    '}}\n'
)


def _compact_prompt_value(value, max_chars: int = 6000) -> str:
    if value is None:
        return 'Not available'
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + '\n...[truncated]'


def planner_prompt(
    goal: str,
    screen_html: str,
    memory: list[str] | str | None = None,
) -> str:
    if isinstance(memory, list):
        memory = '\n'.join(memory) if memory else 'No prior execution memory.'
    return PLANNER_PROMPT_TEMPLATE.format(
        goal=goal,
        screen_html=_compact_prompt_value(screen_html),
        memory=_compact_prompt_value(memory),
    )


def replan_prompt(
    goal: str,
    previous_plan,
    completed_subgoals,
    failed_subgoal,
    expected_outcome: str,
    memory: list[str] | str | None,
    screen_html: str,
) -> str:
    if isinstance(memory, list):
        memory = '\n'.join(memory) if memory else 'No execution memory.'
    return REPLAN_PROMPT_TEMPLATE.format(
        goal=goal,
        previous_plan=_compact_prompt_value(previous_plan, max_chars=4000),
        completed_subgoals=_compact_prompt_value(completed_subgoals, max_chars=3000),
        failed_subgoal=_compact_prompt_value(failed_subgoal, max_chars=2000),
        expected_outcome=expected_outcome or 'Not available',
        memory=_compact_prompt_value(memory, max_chars=5000),
        screen_html=_compact_prompt_value(screen_html),
    )


def subgoal_aware_goal(
    overall_goal: str,
    subgoal: str,
    expected_outcome: str,
) -> str:
    return (
        'Overall user goal:\n'
        f'{overall_goal}\n\n'
        'Current subgoal:\n'
        f'{subgoal}\n\n'
        'Expected outcome for this subgoal:\n'
        f'{expected_outcome}\n\n'
        'Important:\n'
        '- Judge whether the action helps complete the current subgoal.\n'
        '- The action must also remain consistent with the overall user goal.\n'
        '- The action {"action_type": "status", "goal_status": "complete"} is helpful only if the expected outcome is already satisfied.\n'
    )


def action_completion_prompt(
    goal: str,
    action: str,
    history: list[str],
    ui_desc: str,
) -> str:
    """generate the prompt for the action completion.

    Args:
        goal: The current goal.
        history: Summaries for previous steps.
        ui_desc: A list of descriptions for the UI elements.

    Returns:
        The text prompt for action selection that will be sent to gpt4v.
    """
    if history:
        history = '\n'.join(history)
    else:
        history = 'You just started, no action has been performed yet.'

    return ACTION_COMPLETION_PROMPT_TEMPLATE.format(
        goal=goal,
        action=action,
        history=history,
        ui_desc=ui_desc if ui_desc else 'Not available',
    )


def summarize_prompt(
    action: str,
    reason: str,
    goal: str,
    before_elements: str,
    after_elements: str,
    input_type: str = "html",
) -> str:
    """Generate the prompt for the summarization step.

    Args:
        action: Action picked.
        reason: The reason to pick the action.
        goal: The overall goal.
        before_elements: Information for UI elements on the before screenshot.
        after_elements: Information for UI elements on the after screenshot.

    Returns:
        The text prompt for summarization that will be sent to gpt4v.
    """
    return SUMMARY_PROMPT_TEMPLATE_VERIFIER.format(
        goal=goal,
        before_elements=before_elements,
        after_elements=after_elements,
        action=action,
    )


def action_selection_prompt_with_verifier(
    action: str,
    history: list[str],
    goal: str,
    before_elements: str,
) -> str:
    """Generate the prompt for the self-evaluation step.

    Args:
        action: Action picked.
        reason: The reason to pick the action.
        goal: The overall goal.
        before_elements: Information for UI elements on the before screenshot.

    Returns:
        The text prompt for summarization that will be sent to gpt4v.
    """

    if history:
        history = '\n'.join(history)
    else:
        history = 'You just started, no action has been performed yet.'

    prompt_prefix = PROMPT_PREFIX_V2
    return SELF_EVAL_TEMPLATE_VERIFIER_TRAINING_V3.format(
        prompt_prefix=prompt_prefix,
        goal=goal,
        history=history,
        before_elements=before_elements,
        action=action,
    )
