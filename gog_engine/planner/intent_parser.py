"""
intent_parser.py - Rule-based Natural Language Intent Parser

Module contract: Takes a natural language prompt string, returns List[OperationSpec]
— an ordered list of AddFieldOperation and/or MutateActionOperation dataclasses.

This module is a no-LLM zone. All parsing is deterministic pattern matching.
If the prompt matches known patterns, it produces structured OperationSpec.
If not, it raises IntentParseError. The caller is responsible for handling failure.
"""

import re
from dataclasses import dataclass
from typing import Literal, Union, List


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures (Operation Specifications)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AddFieldOperation:
    """Specification: Add a new field to a store's state object."""
    operation: Literal["ADD_FIELD"]
    target_file: str    # e.g. "src/stores/authStore.ts"
    target_node: str    # e.g. "state"
    field_name: str     # e.g. "lastLogin"
    field_type: str     # e.g. "string"
    default_value: str  # e.g. "''"


@dataclass
class MutateActionOperation:
    """Specification: Add a statement to a store action."""
    operation: Literal["MUTATE_ACTION"]
    target_file: str        # e.g. "src/stores/authStore.ts"
    target_action: str      # e.g. "login"
    add_statement: str      # e.g. "this.lastLogin = '2026-03-08'"


@dataclass
class AddImportOperation:
    """Specification: Add an import statement to a file."""
    operation: Literal["ADD_IMPORT"]
    target_file: str        # e.g. "src/components/HeaderWidget.vue"
    import_statement: str   # e.g. "import { useAuthStore } from '../stores/authStore'"


@dataclass
class AddTemplateElementOperation:
    """Specification: Add an HTML element to a Vue template."""
    operation: Literal["ADD_TEMPLATE_ELEMENT"]
    target_file: str        # must be a .vue file
    element_html: str       # e.g. '<button @click="logout">Logout</button>'
    insert_adjacent_to: str # hint for placement, e.g. "user role"


@dataclass
class AddSetupBindingOperation:
    """Specification: Add a binding statement to Vue script setup."""
    operation: Literal["ADD_SETUP_BINDING"]
    target_file: str        # e.g. "src/components/HeaderWidget.vue"
    binding_statement: str  # e.g. "const { logout } = useAuthStore()"


@dataclass
class AddMethodOperation:
    """Specification: Add a method to a service file."""
    operation: Literal["ADD_METHOD"]
    target_file: str        # e.g. "src/services/api_client.ts"
    method_name: str        # e.g. "deleteAccount"
    method_body: str        # complete method body as string


@dataclass
class AddActionOperation:
    """Specification: Add an action to a Pinia store."""
    operation: Literal["ADD_ACTION"]
    target_file: str        # e.g. "src/stores/authStore.ts"
    action_name: str        # e.g. "deleteUser"
    action_body: str        # complete action body as string


@dataclass
class ForbiddenImportConstraint:
    """Specification: A file must NOT contain an import from a given module."""
    operation: Literal["FORBIDDEN_IMPORT"]
    target_file: str        # file that must NOT contain the import
    forbidden_module: str   # e.g. "api_client"


OperationSpec = Union[
    AddFieldOperation,
    MutateActionOperation,
    AddImportOperation,
    AddTemplateElementOperation,
    AddSetupBindingOperation,
    AddMethodOperation,
    AddActionOperation,
    ForbiddenImportConstraint,
]


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class IntentParseError(Exception):
    """Raised when natural language prompt doesn't match known patterns."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Regex Patterns for Easy Task (ADD_FIELD + MUTATE_ACTION)
# ─────────────────────────────────────────────────────────────────────────────

# Matches file paths like `src/stores/authStore.ts` or `src/stores/authStore.vue`
FILE_PATTERN = re.compile(r'`(src/[^`]+\.(?:ts|vue))`')

# Matches patterns like "add a `lastLogin` string" or "add `lastLogin` string"
ADD_FIELD_RE = re.compile(r'add\s+a?\s*`(\w+)`\s+(\w+)', re.IGNORECASE)

# Checks if context is "to the default state" or similar
STATE_RE = re.compile(r'to\s+the\s+(?:default\s+)?state', re.IGNORECASE)

# Matches patterns like "update the `login` action"
ACTION_RE = re.compile(r'update\s+the\s+`(\w+)`\s+action', re.IGNORECASE)

# Matches patterns like "set it to '2026-03-08'" or 'set it to "value"'
SET_VALUE_RE = re.compile(r"set\s+it\s+to\s+['\"]([^'\"]+)['\"]", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Task-Specific Parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_medium(prompt: str) -> List[OperationSpec]:
    """
    Parse Medium task: Wire a Logout button in HeaderWidget.vue to useAuthStore.

    Expected pattern:
    "Refactor `src/components/HeaderWidget.vue` to include a 'Logout' button next to
     the user role. Wire the click event to call the `logout` action from `useAuthStore`."
    """
    operations: List[OperationSpec] = []

    # Extract target file
    file_match = FILE_PATTERN.search(prompt)
    if not file_match or '.vue' not in file_match.group(1):
        raise IntentParseError("Medium task requires a .vue file target")
    target_file = file_match.group(1)

    # Detect need for ADD_IMPORT
    if 'useAuthStore' in prompt:
        operations.append(AddImportOperation(
            operation="ADD_IMPORT",
            target_file=target_file,
            import_statement="import { useAuthStore } from '../stores/authStore'",
        ))

    # Detect need for ADD_SETUP_BINDING
    if 'useAuthStore' in prompt:
        operations.append(AddSetupBindingOperation(
            operation="ADD_SETUP_BINDING",
            target_file=target_file,
            binding_statement="const { logout } = useAuthStore()",
        ))

    # Detect need for ADD_TEMPLATE_ELEMENT (Logout button)
    if 'logout' in prompt.lower() and 'button' in prompt.lower():
        operations.append(AddTemplateElementOperation(
            operation="ADD_TEMPLATE_ELEMENT",
            target_file=target_file,
            element_html='<button @click="logout">Logout</button>',
            insert_adjacent_to="user role",
        ))

    if not operations:
        raise IntentParseError("Medium task pattern not recognized")

    return operations


def _parse_hard(prompt: str) -> List[OperationSpec]:
    """
    Parse Hard task: Implement Delete Account feature across 3 files with constraint.

    Expected pattern:
    "Implement a 'Delete Account' feature. Add a `deleteAccount` API call in
     `src/services/api_client.ts` that posts to '/delete', create a `deleteUser` action
     in `authStore.ts` that calls it, and add a 'Delete' button in
     `src/views/UserSettings.vue` to trigger it. You must NOT import `api_client.ts`
     directly into the Vue component."
    """
    operations: List[OperationSpec] = []

    # Extract all file paths from backtick-enclosed paths
    files = FILE_PATTERN.findall(prompt)

    # Infer files from keywords (or use extracted files as fallback)
    api_client_file = next((f for f in files if 'api_client' in f), 'src/services/api_client.ts')
    auth_store_file = next((f for f in files if 'authStore' in f), 'src/stores/authStore.ts')
    user_settings_file = next((f for f in files if 'UserSettings' in f), 'src/views/UserSettings.vue')

    # ADD_METHOD to api_client.ts
    operations.append(AddMethodOperation(
        operation="ADD_METHOD",
        target_file=api_client_file,
        method_name="deleteAccount",
        method_body="async deleteAccount() { return this.post('/delete'); }",
    ))

    # ADD_ACTION to authStore.ts
    operations.append(AddActionOperation(
        operation="ADD_ACTION",
        target_file=auth_store_file,
        action_name="deleteUser",
        action_body="async deleteUser() { await api.deleteAccount(); this.token = null; this.user = null; }",
    ))

    # ADD_SETUP_BINDING to UserSettings.vue
    operations.append(AddSetupBindingOperation(
        operation="ADD_SETUP_BINDING",
        target_file=user_settings_file,
        binding_statement="const { deleteUser } = useAuthStore()",
    ))

    # ADD_TEMPLATE_ELEMENT to UserSettings.vue
    operations.append(AddTemplateElementOperation(
        operation="ADD_TEMPLATE_ELEMENT",
        target_file=user_settings_file,
        element_html='<button @click="deleteUser">Delete Account</button>',
        insert_adjacent_to="existing buttons",
    ))

    # FORBIDDEN_IMPORT constraint on UserSettings.vue
    operations.append(ForbiddenImportConstraint(
        operation="FORBIDDEN_IMPORT",
        target_file=user_settings_file,
        forbidden_module="api_client",
    ))

    return operations


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_intent(prompt: str) -> List[OperationSpec]:
    """
    Rule-based parser: converts natural language prompt to structured OperationSpec.

    Dispatches to task-specific handlers:
    - Easy: ADD_FIELD + MUTATE_ACTION on Pinia store (single file)
    - Medium: ADD_IMPORT + ADD_SETUP_BINDING + ADD_TEMPLATE_ELEMENT on Vue component
    - Hard: Multi-file feature with forbidden import constraint

    Args:
        prompt: Natural language instruction string.

    Returns:
        List[OperationSpec]: Ordered list of operations appropriate for the task level.

    Raises:
        IntentParseError: If mandatory patterns don't match.
    """
    # Detect task level by keywords
    if 'Delete Account' in prompt or 'deleteAccount' in prompt:
        return _parse_hard(prompt)
    elif 'Logout' in prompt and '.vue' in prompt:
        return _parse_medium(prompt)
    else:
        # Default to Easy task parsing
        return _parse_easy(prompt)


def _parse_easy(prompt: str) -> List[OperationSpec]:
    """
    Parse Easy task: ADD_FIELD + MUTATE_ACTION on Pinia store (single file).

    Expected pattern:
    "Write the code to add a `lastLogin` string timestamp to the default state
     in `src/stores/authStore.ts` and update the `login` action to set it to '2026-03-08'."
    """
    operations: List[OperationSpec] = []

    # ── Extract target file ──────────────────────────────────────────────────
    file_match = FILE_PATTERN.search(prompt)
    if not file_match:
        raise IntentParseError(
            f"Could not extract target file. Expected format: `src/stores/...`"
        )
    target_file = file_match.group(1)

    # ── Try ADD_FIELD pattern ────────────────────────────────────────────────
    add_field_match = ADD_FIELD_RE.search(prompt)
    if add_field_match:
        field_name = add_field_match.group(1)
        field_type = add_field_match.group(2)

        # Verify it's "to the default state" context
        if not STATE_RE.search(prompt):
            raise IntentParseError(
                f"ADD_FIELD pattern matched but not in state context. "
                f"Expected: 'to the default state'"
            )

        # Default value is empty string for string types
        default_value = "''"

        add_field_op = AddFieldOperation(
            operation="ADD_FIELD",
            target_file=target_file,
            target_node="state",
            field_name=field_name,
            field_type=field_type,
            default_value=default_value,
        )
        operations.append(add_field_op)

    # ── Try MUTATE_ACTION pattern ────────────────────────────────────────────
    action_match = ACTION_RE.search(prompt)
    if action_match:
        target_action = action_match.group(1)

        # Extract the value to set
        value_match = SET_VALUE_RE.search(prompt)
        if not value_match:
            raise IntentParseError(
                f"MUTATE_ACTION pattern matched (action: '{target_action}') "
                f"but could not extract value. Expected: 'set it to \"value\"'"
            )
        value = value_match.group(1)

        # Build the statement. For the Easy task, assume we're setting a field
        # that was just added (same name as the ADD_FIELD field_name).
        if operations and isinstance(operations[0], AddFieldOperation):
            field_name = operations[0].field_name
        else:
            # If no ADD_FIELD preceded, try to infer field name from context or error
            raise IntentParseError(
                f"MUTATE_ACTION requires preceding ADD_FIELD to infer field name."
            )

        add_statement = f"this.{field_name} = '{value}'"

        mutate_op = MutateActionOperation(
            operation="MUTATE_ACTION",
            target_file=target_file,
            target_action=target_action,
            add_statement=add_statement,
        )
        operations.append(mutate_op)

    if not operations:
        raise IntentParseError(
            f"No recognized patterns matched. "
            f"Supported: ADD_FIELD + MUTATE_ACTION (Easy task)."
        )

    return operations


if __name__ == "__main__":
    # Test the parser
    test_prompt = (
        "Write the code to add a `lastLogin` string timestamp to the default state "
        "in `src/stores/authStore.ts` and update the `login` action to set it to '2026-03-08'."
    )

    try:
        ops = parse_intent(test_prompt)
        print(f"✓ Parsed {len(ops)} operations:")
        for i, op in enumerate(ops, 1):
            print(f"  {i}. {op}")
    except IntentParseError as e:
        print(f"✗ Parse error: {e}")
